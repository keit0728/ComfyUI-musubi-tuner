# Wan2.1推論カスタムノード実装方針

## 概要

Wan2.1の動画生成推論を実行するComfyUIカスタムノードの実装方針をまとめます。musubi-tunerのコード改変は不可という制約のもと、CLIコマンドをラップする形で実装します。

## 実装アーキテクチャ

### 基本方針
- musubi-tunerの仮想環境でCLIコマンドを実行
- 標準出力/エラー出力のパースによる進捗表示とエラーハンドリング
- ファイルベースでの入出力管理

### ノード構成

#### 1. MusubiTuner_Wan21_VideoGenerator
メインの推論実行ノード。`wan_generate_video.py`をラップします。

## 入力パラメータ設計

### 必須パラメータ

| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| task | COMBO | "t2v-14B" | タスクタイプ（後述） |
| prompt | STRING | "" | 生成プロンプト |
| save_path | STRING | "" | 出力ファイルパス |
| dit | STRING | "" | DiTモデルパス |
| vae | STRING | "" | VAEモデルパス |
| t5 | STRING | "" | T5エンコーダパス |

### タスクタイプ
```python
TASK_CHOICES = [
    "t2v-1.3B",     # Text to Video 1.3B
    "t2v-14B",      # Text to Video 14B
    "i2v-14B",      # Image to Video 14B
    "t2i-14B",      # Text to Image 14B
    "flf2v-14B",    # First/Last Frame to Video 14B
    "t2v-1.3B-FC",  # Fun Control T2V 1.3B
    "t2v-14B-FC",   # Fun Control T2V 14B
    "i2v-14B-FC",   # Fun Control I2V 14B
]
```

### オプションパラメータ

#### 生成設定
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| video_size | INT_PAIR | [832, 480] | 動画サイズ（タスク別制約あり） |
| video_length | INT | 81 | 動画の長さ（フレーム数） |
| fps | INT | 16 | フレームレート |
| infer_steps | INT | 20 | 推論ステップ数 |
| seed | INT | -1 | シード値（-1でランダム） |
| negative_prompt | STRING | "" | ネガティブプロンプト |
| guidance_scale | FLOAT | 5.0 | CFGスケール |
| flow_shift | FLOAT | -1 | Flow shift（-1で自動） |

#### 性能・メモリ設定
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| fp8 | BOOLEAN | False | fp8モード有効化 |
| fp8_scaled | BOOLEAN | False | fp8スケーリング有効化 |
| fp8_fast | BOOLEAN | False | fp8高速モード（RTX 40x0） |
| fp8_t5 | BOOLEAN | False | T5のfp8モード |
| blocks_to_swap | INT | 0 | スワップブロック数 |
| attn_mode | COMBO | "torch" | アテンションモード |
| vae_cache_cpu | BOOLEAN | False | VAEキャッシュをCPU使用 |

#### I2V/V2V関連
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| image_path | STRING | "" | 初期画像パス（I2V用） |
| end_image_path | STRING | "" | 終了画像パス（flf2v用） |
| video_path | STRING | "" | 入力動画パス（V2V用） |
| clip | STRING | "" | CLIPモデルパス（I2V必須） |

#### Fun Control関連
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| control_path | STRING | "" | 制御動画パス |

#### LoRA関連
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| lora_weight | STRING | "" | LoRAファイルパス（複数可） |
| lora_multiplier | FLOAT | 1.0 | LoRA適用強度（複数可） |

#### 高度な設定
| パラメータ名 | 型 | デフォルト | 説明 |
|------------|---|----------|------|
| cfg_skip_mode | COMBO | "none" | CFGスキップモード |
| cfg_apply_ratio | FLOAT | 1.0 | CFG適用率 |
| trim_tail_frames | INT | 0 | 末尾フレームのトリム数 |
| output_type | COMBO | "video" | 出力タイプ |
| cpu_noise | BOOLEAN | False | CPUでノイズ生成 |

### アテンションモード
```python
ATTN_MODES = ["torch", "sdpa", "xformers", "sageattn", "flash", "flash2", "flash3"]
```

### CFGスキップモード
```python
CFG_SKIP_MODES = ["none", "early", "late", "middle", "early_late", "alternate"]
```

### 出力タイプ
```python
OUTPUT_TYPES = ["video", "images", "latent", "both", "latent_images"]
```

## 実装詳細

### 1. パラメータバリデーション

```python
def validate_inputs(self, **kwargs):
    """入力パラメータの検証"""
    
    # タスクとビデオサイズの整合性チェック
    task = kwargs['task']
    video_size = kwargs['video_size']
    
    # SUPPORTED_SIZESから有効なサイズを取得
    supported_sizes = {
        "t2v-14B": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
        "t2v-1.3B": [(480, 832), (832, 480)],
        # ... 他のタスクも同様
    }
    
    if tuple(video_size) not in supported_sizes.get(task, []):
        raise ValueError(f"Invalid video size {video_size} for task {task}")
    
    # I2Vタスクの場合はCLIPパスが必須
    if "i2v" in task and not kwargs.get('clip'):
        raise ValueError("CLIP model path is required for I2V tasks")
    
    # Fun Controlタスクの場合は制御パスが必須
    if "FC" in task and not kwargs.get('control_path'):
        raise ValueError("Control path is required for Fun Control tasks")
```

### 2. コマンド構築

```python
def build_command(self, **kwargs):
    """CLIコマンドの構築"""
    
    cmd_args = [
        "src/musubi_tuner/wan_generate_video.py",
        "--task", kwargs['task'],
        "--prompt", kwargs['prompt'],
        "--save_path", kwargs['save_path'],
        "--dit", kwargs['dit'],
        "--vae", kwargs['vae'],
        "--t5", kwargs['t5'],
        "--video_size", str(kwargs['video_size'][0]), str(kwargs['video_size'][1]),
        "--video_length", str(kwargs['video_length']),
        "--fps", str(kwargs['fps']),
        "--infer_steps", str(kwargs['infer_steps']),
        "--attn_mode", kwargs['attn_mode'],
        "--output_type", kwargs['output_type'],
    ]
    
    # オプションパラメータの追加
    if kwargs['seed'] >= 0:
        cmd_args.extend(["--seed", str(kwargs['seed'])])
    
    if kwargs['negative_prompt']:
        cmd_args.extend(["--negative_prompt", kwargs['negative_prompt']])
    
    if kwargs['fp8']:
        cmd_args.append("--fp8")
    
    if kwargs['fp8_scaled']:
        cmd_args.append("--fp8_scaled")
    
    if kwargs['blocks_to_swap'] > 0:
        cmd_args.extend(["--blocks_to_swap", str(kwargs['blocks_to_swap'])])
    
    # LoRA設定
    if kwargs['lora_weight']:
        lora_weights = kwargs['lora_weight'].split(',')
        lora_multipliers = str(kwargs['lora_multiplier']).split(',')
        
        cmd_args.append("--lora_weight")
        cmd_args.extend(lora_weights)
        cmd_args.append("--lora_multiplier")
        cmd_args.extend(lora_multipliers)
    
    return cmd_args
```

### 3. 進捗パース

```python
def parse_progress(self, line):
    """標準出力から進捗情報を抽出"""
    
    # Sampling loop time stepのパターン
    # 例: "Sampling loop time step: 3/20"
    sampling_match = re.search(r'Sampling loop time step:\s*(\d+)/(\d+)', line)
    if sampling_match:
        current = int(sampling_match.group(1))
        total = int(sampling_match.group(2))
        return {
            'stage': 'sampling',
            'progress': current / total,
            'message': f"Sampling: {current}/{total}"
        }
    
    # VAEデコードのパターン
    # 例: "Decoding latents: 100%|████| 10/10"
    vae_match = re.search(r'Decoding latents:\s*(\d+)%', line)
    if vae_match:
        progress = int(vae_match.group(1)) / 100
        return {
            'stage': 'vae_decode',
            'progress': progress,
            'message': f"VAE Decoding: {int(progress * 100)}%"
        }
    
    # モデルロード
    if "Loading" in line and "model" in line:
        return {
            'stage': 'loading',
            'progress': None,
            'message': "Loading models..."
        }
    
    return None
```

### 4. エラーハンドリング

```python
def handle_error(self, stderr, returncode):
    """エラーメッセージの解析と適切な例外の発生"""
    
    error_patterns = {
        "CUDA out of memory": "GPU メモリ不足です。以下を試してください:\n" +
                             "- fp8モードを有効にする\n" +
                             "- blocks_to_swapを増やす（最大39）\n" +
                             "- 解像度やvideo_lengthを減らす",
        
        "No such file or directory": "ファイルが見つかりません。パスを確認してください。",
        
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied": 
            "モデルとタスクの組み合わせが間違っています。",
        
        "Invalid video size": "指定されたビデオサイズはこのタスクでサポートされていません。",
        
        "CLIP model path is required": "I2VタスクにはCLIPモデルパスが必要です。"
    }
    
    for pattern, message in error_patterns.items():
        if pattern in stderr:
            raise MusubiTunerError(message)
    
    # デフォルトエラー
    raise MusubiTunerError(f"実行エラー (code: {returncode}):\n{stderr}")
```

## ComfyUIノード実装

```python
class MusubiTuner_Wan21_VideoGenerator:
    """Wan2.1動画生成ノード"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (TASK_CHOICES, {"default": "t2v-14B"}),
                "prompt": ("STRING", {"multiline": True}),
                "save_path": ("STRING", {"default": "output.mp4"}),
                "dit": ("STRING", {"default": ""}),
                "vae": ("STRING", {"default": ""}),
                "t5": ("STRING", {"default": ""}),
                "video_size": ("INT", {
                    "default": [832, 480],
                    "min": 256,
                    "max": 2048,
                    "step": 16
                }),
                "video_length": ("INT", {"default": 81, "min": 1, "max": 200}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
                "infer_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "flow_shift": ("FLOAT", {"default": -1.0}),
                "fp8": ("BOOLEAN", {"default": False}),
                "fp8_scaled": ("BOOLEAN", {"default": False}),
                "fp8_fast": ("BOOLEAN", {"default": False}),
                "fp8_t5": ("BOOLEAN", {"default": False}),
                "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 39}),
                "attn_mode": (ATTN_MODES, {"default": "torch"}),
                "vae_cache_cpu": ("BOOLEAN", {"default": False}),
                "image_path": ("STRING", {"default": ""}),
                "end_image_path": ("STRING", {"default": ""}),
                "video_path": ("STRING", {"default": ""}),
                "clip": ("STRING", {"default": ""}),
                "control_path": ("STRING", {"default": ""}),
                "lora_weight": ("STRING", {"default": ""}),
                "lora_multiplier": ("FLOAT", {"default": 1.0}),
                "cfg_skip_mode": (CFG_SKIP_MODES, {"default": "none"}),
                "cfg_apply_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "trim_tail_frames": ("INT", {"default": 0, "min": 0}),
                "output_type": (OUTPUT_TYPES, {"default": "video"}),
                "cpu_noise": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "log")
    FUNCTION = "generate_video"
    CATEGORY = "MusubiTuner/Wan2.1"
    
    def generate_video(self, **kwargs):
        """動画生成の実行"""
        
        # 入力検証
        self.validate_inputs(**kwargs)
        
        # Executorインスタンス作成
        executor = MusubiTunerExecutor()
        
        # コマンド構築
        cmd_args = self.build_command(**kwargs)
        
        # 進捗コールバック
        progress_info = {"stage": "", "progress": 0}
        
        def update_progress(info):
            if info:
                progress_info.update(info)
                # ComfyUIの進捗更新API呼び出し
                # （実装はComfyUIのバージョンに依存）
        
        # 実行
        try:
            output = executor.execute_command(
                cmd_args, 
                progress_callback=lambda line: update_progress(self.parse_progress(line))
            )
            
            # 出力パスの確認
            output_path = kwargs['save_path']
            if not os.path.exists(output_path):
                raise MusubiTunerError(f"出力ファイルが生成されませんでした: {output_path}")
            
            return (output_path, output)
            
        except Exception as e:
            # エラーハンドリング
            if hasattr(e, 'stderr'):
                self.handle_error(e.stderr, e.returncode)
            raise
```

## 設定管理

ComfyUIの設定ファイルで管理：

```json
{
  "musubi_tuner": {
    "wan21": {
      "default_negative_prompt": "low quality, blurry, distorted",
      "default_flow_shift": {
        "480p_i2v": 3.0,
        "default": 5.0
      },
      "max_blocks_to_swap": {
        "14B": 39,
        "1.3B": 29
      },
      "progress_update_interval": 0.5
    }
  }
}
```

## 今後の拡張

### Phase 1（基本実装）
- 基本的なT2V推論
- 進捗表示
- エラーハンドリング

### Phase 2（機能拡張）
- I2V/V2V対応
- LoRA適用
- Fun Control対応
- バッチ処理

### Phase 3（最適化）
- キャッシュ機能
- プリセット管理
- ワークフロー例の提供

## まとめ

このドキュメントに基づいて実装することで、Wan2.1の推論機能をComfyUIから利用できるようになります。musubi-tunerのコード改変なしに、CLIラッパーとして実装することで、安定した動作と容易なメンテナンスを実現します。