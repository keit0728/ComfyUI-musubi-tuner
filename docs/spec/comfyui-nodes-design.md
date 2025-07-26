# Musubi Tuner ComfyUIカスタムノード設計方針

## 概要

Musubi TunerをComfyUIのカスタムノードとして実装する際の設計方針をまとめます。ComfyUIのノードベース設計の利点を活かし、ユーザーが直感的に動画生成AIのLoRA学習・推論ワークフローを構築できるようにします。

## 設計原則

1. **単一責任の原則**: 各ノードは明確に定義された1つの機能を持つ
2. **再利用性**: 異なるモデル間で共通の機能は共有ノードとして実装
3. **モジュラー性**: ノードを自由に組み合わせて複雑なワークフローを構築可能
4. **段階的実行**: 重い処理（キャッシュ、学習）の結果を保存し、再利用可能に
5. **エラーハンドリング**: 各ノードで適切なバリデーションとエラーメッセージを提供

## ノードカテゴリと機能分割

各ノードは実装方式により以下に分類されます：
- 🔧 **CLIラッパー**: 既存のCLIコマンドをラップして実装（別環境実行可能）
- 🆕 **新規実装**: Python APIを直接使用して新規実装が必要
- ❌ **実装困難**: 別環境実行では実装が困難または不可能
- ⚠️ **制約あり**: 別環境実行でも可能だが制約や制限がある

### 1. データ準備ノード群

#### 1.1 データセット管理
- **Dataset Config Loader** 🆕
  - TOML設定ファイルの読み込み
  - データセット情報の解析とバリデーション
  - 出力: データセット設定オブジェクト
  - **実装方式**: Python APIで新規実装（TOMLパーサーとGUI統合）

- **Dataset Viewer** 🆕
  - データセットの内容を可視化
  - 画像/動画とキャプションのプレビュー
  - デバッグ用途
  - **実装方式**: Python APIで新規実装（可視化機能）

#### 1.2 キャッシュノード
- **VAE Latent Cache** 🔧
  - 画像/動画をVAEでエンコードしてキャッシュ
  - モデルタイプ（HunyuanVideo/Wan/FramePack）に対応
  - 出力: キャッシュパス情報
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    # HunyuanVideo用
    python src/musubi_tuner/cache_latents.py \
      --dataset_config path/to/toml \
      --vae path/to/vae/pytorch_model.pt \
      --vae_chunk_size 32 --vae_tiling
    
    # FramePack用
    python src/musubi_tuner/fpack_cache_latents.py \
      --dataset_config path/to/toml \
      --vae path/to/vae \
      --vae_tiling
    
    # Wan用
    python src/musubi_tuner/wan_cache_latents.py \
      --dataset_config path/to/toml \
      --vae path/to/vae
    ```

- **Text Encoder Cache** 🔧
  - テキストプロンプトのエンコード結果をキャッシュ
  - 複数のText Encoder（CLIP、LLM等）に対応
  - 出力: キャッシュパス情報
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    # HunyuanVideo用
    python src/musubi_tuner/cache_text_encoder_outputs.py \
      --dataset_config path/to/toml \
      --text_encoder1 path/to/text_encoder \
      --text_encoder2 path/to/text_encoder_2 \
      --batch_size 16
    
    # FramePack用
    python src/musubi_tuner/fpack_cache_text_encoder_outputs.py \
      --dataset_config path/to/toml \
      --text_encoder path/to/text_encoder
    
    # Wan用
    python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
      --dataset_config path/to/toml \
      --llm_path path/to/llm \
      --batch_size 8
    ```

### 2. 学習ノード群

#### 2.1 学習設定ノード
- **LoRA Training Config** 🆕
  - 学習パラメータの設定
  - network_dim、learning_rate、epochs等
  - 省メモリオプション（fp8、block_swap等）の設定
  - **実装方式**: Python APIで新規実装（GUI設定）

- **Optimizer Config** 🆕
  - 最適化アルゴリズムの選択と設定
  - adamw8bit系の各種オプション
  - **実装方式**: Python APIで新規実装（GUI設定）

- **Sampling Config** 🆕
  - timestep samplingの設定
  - discrete flow shiftの調整
  - サンプル生成設定
  - **実装方式**: Python APIで新規実装（GUI設定）

#### 2.2 モデル別学習ノード
- **HunyuanVideo LoRA Trainer** 🔧
  - HunyuanVideo専用のLoRA学習
  - 入力: モデル、データセット、キャッシュ、設定
  - 出力: 学習済みLoRAファイル
  - **実装方式**: CLIラッパー（accelerate launch経由）
  - **CLIコマンド例**:
    ```bash
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
      src/musubi_tuner/hv_train_network.py \
      --dit path/to/transformers/mp_rank_00_model_states.pt \
      --dataset_config path/to/toml \
      --sdpa --mixed_precision bf16 --fp8_base \
      --optimizer_type adamw8bit --learning_rate 2e-4 \
      --gradient_checkpointing \
      --network_module networks.lora --network_dim 32 \
      --timestep_sampling shift --discrete_flow_shift 7.0 \
      --max_train_epochs 16 --save_every_n_epochs 1 \
      --output_dir path/to/output_dir \
      --output_name name-of-lora
    ```

- **Wan2.1 LoRA Trainer** 🔧
  - Wan2.1専用のLoRA学習
  - T2V/I2V/Fun-Controlモード対応
  - **実装方式**: CLIラッパー（accelerate launch経由）
  - **CLIコマンド例**:
    ```bash
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
      src/musubi_tuner/wan_train_network.py \
      --config_file wan_t2v_config.py \
      --dataset_config path/to/toml \
      --output_dir path/to/output \
      --output_name wan_lora \
      --learning_rate 1e-4 \
      --train_batch_size 1 \
      --max_train_epochs 10
    ```

- **FramePack LoRA Trainer** 🔧
  - FramePack専用のLoRA学習
  - I2V特化の設定オプション
  - **実装方式**: CLIラッパー（accelerate launch経由）
  - **CLIコマンド例**:
    ```bash
    accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
      src/musubi_tuner/fpack_train_network.py \
      --dit path/to/dit_model.pt \
      --dataset_config path/to/toml \
      --sdpa --mixed_precision bf16 \
      --optimizer_type adamw8bit \
      --learning_rate 1e-4 \
      --network_module networks.lora_framepack \
      --network_dim 32 \
      --max_train_epochs 10 \
      --output_dir path/to/output
    ```

### 3. 推論ノード群

#### 3.1 モデルローダー（別環境実行では実装困難）
- **HunyuanVideo Model Loader** ❌
  - DiTモデル、VAE、Text Encoderの読み込み
  - fp8オプション対応
  - **実装方式**: 別環境実行では困難（Video Generatorに統合推奨）

- **Wan2.1 Model Loader** ❌
  - Wan2.1モデルの読み込み
  - タスクタイプの自動認識
  - **実装方式**: 別環境実行では困難（Video Generatorに統合推奨）

- **FramePack Model Loader** ❌
  - FramePackモデルの読み込み
  - MagCache設定
  - **実装方式**: 別環境実行では困難（Video Generatorに統合推奨）

#### 3.2 LoRA適用ノード（別環境実行では実装困難）
- **LoRA Loader** ❌
  - LoRAファイルの読み込み
  - 複数LoRAの管理
  - **実装方式**: 別環境実行では困難（Video Generatorに統合推奨）

- **LoRA Applier** ❌
  - モデルにLoRAを適用
  - multiplierの調整
  - 複数LoRAのマージ適用
  - **実装方式**: 別環境実行では困難（Video Generatorに統合推奨）

#### 3.3 生成ノード
- **Video Generator (HunyuanVideo)** 🔧
  - プロンプトから動画生成
  - T2V/I2V/V2V対応
  - 各種生成パラメータ調整
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    python src/musubi_tuner/hv_generate_video.py \
      --fp8 --video_size 544 960 --video_length 5 \
      --infer_steps 30 \
      --prompt "A cat walks on the grass, realistic style." \
      --save_path path/to/save/dir --output_type both \
      --dit path/to/transformers/mp_rank_00_model_states.pt \
      --attn_mode sdpa --split_attn \
      --vae path/to/vae/pytorch_model.pt \
      --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
      --text_encoder1 path/to/text_encoder \
      --text_encoder2 path/to/text_encoder_2 \
      --seed 1234 \
      --lora_multiplier 1.0 \
      --lora_weight path/to/lora.safetensors
    ```

- **Video Generator (Wan2.1)** 🔧
  - Wan2.1での動画生成
  - タスク別の設定
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    python src/musubi_tuner/wan_generate_video.py \
      --config_file wan_t2v_config.py \
      --prompt "A beautiful sunset over the ocean" \
      --num_inference_steps 50 \
      --save_path output_video.mp4 \
      --lora_weight path/to/wan_lora.safetensors \
      --lora_multiplier 1.0 \
      --solver uni_pc
    ```

- **Video Generator (FramePack)** 🔧
  - FramePackでの動画生成
  - セクション別プロンプト対応
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    python src/musubi_tuner/fpack_generate_video.py \
      --dit path/to/dit_model.pt \
      --vae path/to/vae \
      --text_encoder path/to/text_encoder \
      --prompt "A flowing river in the forest" \
      --image_input path/to/first_frame.jpg \
      --video_length 49 \
      --save_path output.mp4 \
      --lora_weight path/to/fpack_lora.safetensors
    ```

### 4. ユーティリティノード群

#### 4.1 LoRA管理
- **LoRA Format Converter** 🔧
  - musubi形式↔ComfyUI形式の変換
  - 自動形式検出
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    # Musubi形式から他形式へ変換
    python src/musubi_tuner/convert_lora.py \
      --input path/to/musubi_lora.safetensors \
      --output path/to/comfyui_lora.safetensors \
      --target other
    
    # 他形式からMusubi形式へ変換
    python src/musubi_tuner/convert_lora.py \
      --input path/to/other_lora.safetensors \
      --output path/to/musubi_lora.safetensors \
      --target default
    ```

- **LoRA Merger** 🔧
  - LoRAを元モデルにマージ
  - 複数LoRAの重み付けマージ
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    # 単一LoRAのマージ
    python src/musubi_tuner/merge_lora.py \
      --dit path/to/original_model.pt \
      --lora_weight path/to/lora.safetensors \
      --save_merged_model path/to/merged_model.safetensors \
      --device cuda \
      --lora_multiplier 1.0
    
    # 複数LoRAのマージ
    python src/musubi_tuner/merge_lora.py \
      --dit path/to/original_model.pt \
      --lora_weight path/to/lora1.safetensors path/to/lora2.safetensors \
      --save_merged_model path/to/merged_model.safetensors \
      --device cuda \
      --lora_multiplier 0.7 0.3
    ```

- **LoRA Post-hoc EMA** 🔧
  - 学習後のEMA適用
  - Power Function EMA対応
  - **実装方式**: CLIラッパー
  - **CLIコマンド例**:
    ```bash
    # 基本的なPost-hoc EMA
    python src/musubi_tuner/lora_post_hoc_ema.py \
      --input_lora path/to/original_lora.safetensors \
      --output_lora path/to/ema_lora.safetensors \
      --ema_decay 0.999
    
    # Power Function EMA (sigma_rel使用)
    python src/musubi_tuner/lora_post_hoc_ema.py \
      --input_lora path/to/original_lora.safetensors \
      --output_lora path/to/ema_lora.safetensors \
      --sigma_rel \
      --beta_min 0.1 \
      --beta_max 0.9
    ```

#### 4.2 入出力ノード
- **Video Input** 🆕
  - 動画ファイルの読み込み
  - フレーム抽出オプション
  - **実装方式**: Python APIで新規実装（動画処理）

- **Video Output** 🆕
  - 生成動画の保存
  - 形式選択（MP4、WebM等）
  - **実装方式**: Python APIで新規実装（動画保存）

- **Batch Processor** 🆕
  - 複数プロンプトのバッチ処理
  - 並列生成管理
  - **実装方式**: Python APIで新規実装（バッチ管理）

### 5. 高度な機能ノード

#### 5.1 メモリ最適化
- **Memory Optimizer** 🆕
  - block_swapの動的調整
  - VAEタイリング設定
  - Attention分割設定
  - **実装方式**: Python APIで新規実装（メモリ管理）

#### 5.2 モニタリング
- **Training Monitor** ⚠️
  - 学習進捗の可視化
  - TensorBoardログの表示
  - リアルタイムloss表示
  - **実装方式**: 別環境実行では制約あり（ログファイル経由での更新）
  - **制約**: リアルタイム性が低下（ポーリング間隔に依存）

- **Resource Monitor** ⚠️
  - VRAM使用量の監視
  - 推奨設定の提案
  - **実装方式**: 別環境実行でも可能（nvidia-smi等を使用）
  - **制約**: プロセス単位の正確な計測が困難

## 実装優先順位

### Phase 1: CLIラッパー実装（最優先）
すでにCLIコマンドが存在するため、比較的簡単に実装可能：
1. **キャッシュ系ノード** 🔧
   - VAE Latent Cache（各モデル用）
   - Text Encoder Cache（各モデル用）
2. **生成ノード** 🔧
   - Video Generator（HunyuanVideo/Wan/FramePack）
3. **LoRA管理ノード** 🔧
   - LoRA Format Converter
   - LoRA Merger
   - LoRA Post-hoc EMA

### Phase 2: 学習機能のCLIラッパー
Accelerate経由での実行が必要：
1. **学習ノード** 🔧
   - 各モデルのLoRA Trainer
   - Accelerate設定の自動化

### Phase 3: 新規実装（基本機能）
CLIでは提供されていないが、使い勝手を向上させる機能：
1. **設定管理** 🆕
   - Dataset Config Loader（GUI編集機能）
   - Training Config（パラメータGUI）
2. **入出力** 🆕
   - Video Input/Output
3. **バッチ処理** 🆕
   - Batch Processor

### Phase 4: 新規実装（高度な機能）
実装コストが高いが価値のある機能：
1. **モニタリング** ⚠️
   - Training Monitor（ログファイル監視）
   - Resource Monitor（システムツール利用）
2. **最適化** 🆕
   - Memory Optimizer

### 実装対象外
別環境実行では実装困難なため除外：
- Model Loader系 ❌
- LoRA Loader/Applier ❌

## ワークフロー例

### 基本的なLoRA学習ワークフロー
```
[Dataset Config] → [VAE Cache] → [Text Encoder Cache] → 
[Training Config] → [HunyuanVideo LoRA Trainer] → [LoRA File]
```

### 推論ワークフロー（別環境実行対応版）
```
[Text Prompt] → [Video Generator] → [Video Output]
         ↑            ↑
    [LoRA File]  [Model Paths Config]
```
※ Video Generator内でモデル読み込みとLoRA適用を統合

### 複数LoRAマージワークフロー
```
[LoRA File 1] →┐
[LoRA File 2] →├→ [LoRA Merger] → [Merged LoRA]
[LoRA File 3] →┘
```

### LoRA形式変換ワークフロー
```
[Musubi LoRA] → [LoRA Format Converter] → [ComfyUI LoRA]
```

## 技術的考慮事項

### 1. CLIコマンドのラッピング方針

Musubi TunerはCLIツールとして設計されているため、ComfyUIノード化では以下の方針で実装します：

#### 1.1 実装アプローチ
- **subprocessモジュールを使用したCLI呼び出し**
  - 各ノードからPythonのsubprocessでCLIコマンドを実行
  - 標準出力・エラー出力をキャプチャしてComfyUIに表示
  - ノード内部では非同期処理を使用してUIの応答性を維持
  - ノード間のデータフローは同期的（前のノードの完了を待つ）

- **実行フローの管理**
  ```python
  # 実装イメージ：UIスレッドをブロックしない非同期実行
  async def execute_cli_command(cmd, progress_callback):
      process = await asyncio.create_subprocess_exec(
          *cmd,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE
      )
      
      # UIスレッドで進捗を更新しながら結果を待つ
      while True:
          line = await process.stdout.readline()
          if not line:
              break
          progress_callback(parse_progress(line))
      
      await process.wait()
      return process.returncode
  ```
  - CLIコマンドの実行中もComfyUIは応答可能
  - 次のノードは現在のノードの完了を待って実行される

#### 1.2 引数の管理
```python
# 例: VAE Latent Cacheノードの実装イメージ
class MusubiTuner_VAELatentCache:
    def execute(self, dataset_config, vae_path, vae_chunk_size=32, vae_tiling=True):
        cmd = [
            "python", "src/musubi_tuner/cache_latents.py",
            "--dataset_config", dataset_config,
            "--vae", vae_path,
            "--vae_chunk_size", str(vae_chunk_size)
        ]
        if vae_tiling:
            cmd.append("--vae_tiling")
        
        # subprocessで実行
        result = subprocess.run(cmd, capture_output=True, text=True)
        return (cache_path,)
```

#### 1.3 Python環境の分離戦略

musubi-tunerとComfyUIは異なる依存関係を持つため、環境分離が必要です：

- **推奨方式: 別仮想環境でのPython実行**
  ```python
  # Windows例
  musubi_python = "C:/path/to/musubi-tuner/venv/Scripts/python.exe"
  cmd = [musubi_python, "src/musubi_tuner/cache_latents.py", ...]
  
  # Linux/Mac例
  musubi_python = "/path/to/musubi-tuner/venv/bin/python"
  ```
  - musubi-tunerの仮想環境パスを設定ファイルで管理
  - 環境変数（PYTHONPATH、CUDA_VISIBLE_DEVICES等）の適切な設定

- **代替方式（将来的な拡張）**
  - REST API化：より柔軟な制御とリアルタイム通信
  - Docker化：完全な環境分離
  
詳細は[Python環境分離戦略ドキュメント](./comfyui-python-env-strategy.md)を参照

- **Accelerate設定の自動化**
  - 初回実行時にaccelerateの設定を自動生成
  - ユーザーが手動で`accelerate config`を実行する必要をなくす

#### 1.4 進捗表示とログ
- **リアルタイム出力**
  - CLIの標準出力をリアルタイムでパース
  - tqdmなどの進捗バーをComfyUIのプログレスバーに変換
  - エラーメッセージを適切にフォーマット

- **ログファイルの管理**
  - 各実行のログをファイルに保存
  - エラー時のデバッグを容易に

#### 1.5 CLIコマンドで直接実現できない機能の実装

musubi-tunerのコード改変が不可のため、以下のノードはComfyUI側で独自に実装します：

- **UI/設定管理ノード**
  - TOML設定ファイルの読み込み・解析・可視化
  - ComfyUIのウィジェットを活用したパラメータ設定
  - 標準的なPythonライブラリ（toml等）のみ使用

- **モニタリングノード**
  - 標準出力のパースによる進捗監視
  - ログファイルの定期的な読み取り
  - システムツール（nvidia-smi等）によるリソース監視

注意：musubi-tunerの内部APIは使用できないため、すべての情報は標準出力やファイル経由で取得する必要があります。

#### 1.6 実装上の考慮事項
- **パスの管理**
  - ComfyUIの作業ディレクトリとmusubi-tunerのパスの整合性
  - 相対パスと絶対パスの適切な変換
  - Windowsパスセパレータの処理

- **プロセス管理**
  - 長時間実行される学習プロセスの管理
  - ユーザーによるキャンセル操作への対応
  - プロセスの強制終了とクリーンアップ

- **エラーリカバリ**
  - CLIコマンドの終了コードチェック
  - 部分的な失敗からの復旧（キャッシュファイルの再利用等）
  - 詳細なエラーメッセージの提供

### 2. 非同期処理とワークフロー制御
- **ノード内部の非同期処理**
  - 重い処理（学習、キャッシュ生成）はUIスレッドをブロックしない
  - リアルタイムのプログレスバー更新
  - ユーザーによるキャンセル操作の即座の反映

- **ノード間の同期制御**
  - 各ノードは前のノードの出力を待って実行開始
  - データ依存関係を保持したまま、UI応答性を確保
  - エラー発生時は後続ノードの実行を中止

### 3. メモリ管理
- ノード間でのモデル共有によるメモリ効率化
- 不要なモデルの自動アンロード

### 4. エラーハンドリング
- 各ノードでの入力検証
- わかりやすいエラーメッセージ
- 自動リトライ機能

### 5. 互換性
- 既存のComfyUIノードとの連携
- 標準的なデータ形式の使用

## 実装方式のまとめ

### CLIラッパーで実装可能なノード（🔧）
以下のノードは既存のCLIコマンドをsubprocessで呼び出すことで実装：
- VAE/Text Encoder Cache（全モデル対応）
- LoRA Trainer（HunyuanVideo/Wan/FramePack）
- Video Generator（全モデル対応）
- LoRA Format Converter
- LoRA Merger
- LoRA Post-hoc EMA

**メリット**：
- 実装が簡単で安定
- 別環境実行により依存関係の競合を回避
- musubi-tunerの更新に自動的に追従

**デメリット**：
- プロセス間通信のオーバーヘッド
- 細かい制御が困難

### 別環境実行では実装困難なノード（❌）
以下のノードはメモリ内でのデータ共有が必要なため実装困難：
- Model Loader（全モデル）
- LoRA Loader/Applier

**推奨対応**：
- Video Generatorノードに機能を統合（現在のCLI設計を維持）

### 制約付きで実装可能なノード（⚠️）
以下のノードは実装可能だが制約あり：
- Training Monitor（ログファイル経由での更新）
- Resource Monitor（システムツール経由）

**制約**：
- リアルタイム性の低下
- 精度の制限

### 新規実装が必要だが問題ないノード（🆕）
以下のノードは別環境実行でも問題なく実装可能：
- Dataset Config Loader/Viewer
- Training/Optimizer/Sampling Config
- Video Input/Output
- Batch Processor
- Memory Optimizer

**特徴**：
- ファイルベースの入出力
- プロセス間通信で十分対応可能

## 開発ガイドライン

1. **命名規則**
   - ノード名: `MusubiTuner_機能名`
   - カテゴリ: `MusubiTuner/サブカテゴリ`

2. **ドキュメント**
   - 各ノードに詳細な説明を含める
   - パラメータのツールチップ
   - 使用例の提供

3. **テスト**
   - 単体テスト
   - 統合テスト（ワークフロー全体）
   - エッジケースの処理

4. **パフォーマンス**
   - 大規模データセットでの動作確認
   - メモリリークの防止
   - 処理時間の最適化