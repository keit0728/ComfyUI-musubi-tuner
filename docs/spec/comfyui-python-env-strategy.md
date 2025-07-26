# ComfyUIとmusubi-tunerのPython環境分離戦略

## 問題の背景

ComfyUIとmusubi-tunerは異なる依存関係を持つため、同一のPython環境での動作は困難：
- **ComfyUI**: 独自の依存関係（torch、各種カスタムノード等）
- **musubi-tuner**: accelerate、diffusers、特定バージョンのPyTorch等

## 重要な前提条件

**musubi-tuner側のコードは改変不可**という制約があるため、以下の方針に限定されます：
- 既存のCLIコマンドのみ使用
- 標準出力/エラー出力のパースによる情報取得
- ファイルベースでの状態管理

## 唯一の解決策：別仮想環境でのCLI実行

### 実装方法

```python
import subprocess
import os
import re
import threading
from pathlib import Path

class MusubiTunerExecutor:
    def __init__(self):
        # 設定ファイルから読み込むか、環境変数で指定
        self.musubi_env_path = os.getenv(
            "MUSUBI_TUNER_ENV", 
            "C:/path/to/musubi-tuner/venv"
        )
        self.musubi_root = os.getenv(
            "MUSUBI_TUNER_ROOT",
            "C:/path/to/musubi-tuner"
        )
        
    def execute_command(self, cmd_args, progress_callback=None):
        """musubi-tuner環境でコマンドを実行"""
        
        # Python実行ファイルのパス
        if os.name == 'nt':  # Windows
            python_exe = Path(self.musubi_env_path) / "Scripts" / "python.exe"
        else:  # Linux/Mac
            python_exe = Path(self.musubi_env_path) / "bin" / "python"
        
        # 環境変数の設定
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.musubi_root)
        
        # プロセス起動
        cmd = [str(python_exe)] + cmd_args
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.musubi_root
        )
        
        # 標準出力を監視して進捗を抽出
        if progress_callback:
            self._monitor_progress(process, progress_callback)
        
        # プロセスの完了を待つ
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Command failed: {stderr}")
            
        return stdout
    
    def _monitor_progress(self, process, callback):
        """標準出力から進捗情報を抽出"""
        def read_output():
            for line in iter(process.stdout.readline, ''):
                if line:
                    progress = self._parse_progress_line(line.strip())
                    if progress is not None:
                        callback(progress)
        
        thread = threading.Thread(target=read_output)
        thread.daemon = True
        thread.start()
    
    def _parse_progress_line(self, line):
        """進捗情報をパース"""
        # tqdm形式: 100%|██████████| 100/100 [00:10<00:00,  9.52it/s]
        tqdm_match = re.search(r'(\d+)%\|', line)
        if tqdm_match:
            return int(tqdm_match.group(1)) / 100
        
        # カスタム形式: Processing 10/100
        custom_match = re.search(r'(\d+)/(\d+)', line)
        if custom_match:
            current = int(custom_match.group(1))
            total = int(custom_match.group(2))
            return current / total
        
        return None
```

### accelerate対応

```python
def execute_accelerate_command(self, script_path, script_args, accelerate_args=None):
    """accelerate launch経由でスクリプトを実行"""
    
    # accelerateの実行ファイル
    if os.name == 'nt':
        accelerate_exe = Path(self.musubi_env_path) / "Scripts" / "accelerate.exe"
    else:
        accelerate_exe = Path(self.musubi_env_path) / "bin" / "accelerate"
    
    # デフォルトのaccelerate設定
    default_accelerate_args = [
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16"
    ]
    
    # コマンド構築
    cmd = [str(accelerate_exe), "launch"]
    cmd.extend(accelerate_args or default_accelerate_args)
    cmd.append(script_path)
    cmd.extend(script_args)
    
    return self.execute_command(cmd[1:])  # "launch"以降を渡す
```

## 進捗情報の取得制限

musubi-tunerのコード改変が不可のため、以下の制限があります：

### 取得可能な情報
1. **標準出力のパース**
   - tqdm等のプログレスバー出力
   - print文による進捗表示
   - ログメッセージ

2. **ファイルベースの監視**
   - 出力ファイルの生成状況
   - ログファイルの更新
   - キャッシュファイルの作成

### 取得困難な情報
1. **詳細な内部状態**
   - メモリ使用量（プロセス単位）
   - 正確な処理段階
   - エラーの詳細コンテキスト

2. **リアルタイム性**
   - バッファリングによる遅延
   - 出力頻度に依存

## ComfyUIノードの実装例

```python
class MusubiTunerCacheLatentsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_config": ("STRING", {"multiline": False}),
                "vae_path": ("STRING", {"multiline": False}),
                "vae_chunk_size": ("INT", {"default": 32, "min": 1, "max": 128}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cache_path",)
    FUNCTION = "cache_latents"
    CATEGORY = "MusubiTuner"
    
    def cache_latents(self, dataset_config, vae_path, vae_chunk_size, vae_tiling):
        executor = MusubiTunerExecutor()
        
        # コマンド引数の構築
        cmd_args = [
            "src/musubi_tuner/cache_latents.py",
            "--dataset_config", dataset_config,
            "--vae", vae_path,
            "--vae_chunk_size", str(vae_chunk_size)
        ]
        
        if vae_tiling:
            cmd_args.append("--vae_tiling")
        
        # プログレスコールバック
        def update_progress(progress):
            # ComfyUIのプログレスバー更新
            # （ComfyUIのAPI仕様に依存）
            pass
        
        # 実行
        try:
            output = executor.execute_command(cmd_args, update_progress)
            
            # 出力からキャッシュパスを抽出
            # （実際の出力形式に応じて調整）
            cache_path = self._extract_cache_path(output)
            
            return (cache_path,)
            
        except Exception as e:
            raise Exception(f"Cache creation failed: {str(e)}")
    
    def _extract_cache_path(self, output):
        """標準出力からキャッシュパスを抽出"""
        # 実装は実際の出力形式に依存
        lines = output.strip().split('\n')
        for line in lines:
            if "Cache saved to:" in line:
                return line.split("Cache saved to:")[-1].strip()
        return ""
```

## エラーハンドリング

```python
class MusubiTunerError(Exception):
    """musubi-tuner実行エラー"""
    pass

def handle_cli_error(stderr, returncode):
    """CLIエラーを解析して適切な例外を発生"""
    
    # よくあるエラーパターン
    if "CUDA out of memory" in stderr:
        raise MusubiTunerError("GPU メモリ不足です。batch_sizeを減らすか、--blocks_to_swapを使用してください。")
    
    if "No such file or directory" in stderr:
        raise MusubiTunerError("ファイルが見つかりません。パスを確認してください。")
    
    if "ModuleNotFoundError" in stderr:
        raise MusubiTunerError("必要なモジュールがインストールされていません。musubi-tuner環境を確認してください。")
    
    # デフォルトエラー
    raise MusubiTunerError(f"コマンド実行に失敗しました (code: {returncode}): {stderr}")
```

## 設定管理

```json
{
  "musubi_tuner": {
    "venv_path": "C:/path/to/musubi-tuner/venv",
    "root_path": "C:/path/to/musubi-tuner",
    "default_accelerate_config": {
      "num_cpu_threads_per_process": 1,
      "mixed_precision": "bf16"
    },
    "progress_update_interval": 0.5,
    "command_timeout": 86400
  }
}
```

## まとめ

musubi-tunerのコード改変が不可という制約下では、**別仮想環境でのCLI実行が唯一の現実的な解決策**です。この方法には以下の特徴があります：

### 利点
- 環境の完全な分離により依存関係の競合を回避
- musubi-tunerの更新に影響されない
- 実装がシンプル

### 制限
- 進捗情報は標準出力のパースに限定
- プロセス間通信のオーバーヘッド
- メモリ内でのデータ共有は不可能

これらの制限を理解した上で、CLIコマンドとして提供されている機能を最大限活用することが重要です。