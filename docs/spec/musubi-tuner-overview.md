# Musubi Tuner プロジェクト概要

## 概要

Musubi Tunerは、動画生成AIモデル（HunyuanVideo、Wan2.1、FramePack）のLoRA（Low-Rank Adaptation）学習用のコマンドラインツールです。省メモリに特化した実装により、比較的限られたハードウェアリソースでも動画生成モデルのファインチューニングが可能です。

## サポートされているモデル

### 1. HunyuanVideo
- Tencentが開発した動画生成モデル
- T2V（Text-to-Video）およびV2V（Video-to-Video）に対応
- 高品質な動画生成が可能

### 2. Wan 2.1
- 複数のタスクタイプに対応：
  - T2V（Text-to-Video）
  - I2V（Image-to-Video）
  - Fun-Control（より高度な制御）
- 1フレーム推論・学習にも対応

### 3. FramePack
- フレーム圧縮技術を使用した効率的なモデル
- 主にI2V（Image-to-Video）タスクに特化
- セクション単位でのプロンプト指定など高度な制御が可能
- MagCacheサポートによる効率的な推論

## 主要機能

### 1. データ前処理
- **latentの事前キャッシュ** (`cache_latents.py`)
  - 画像・動画をVAEでエンコードし、潜在表現を保存
  - 学習時の処理速度を大幅に向上
  
- **Text Encoder出力の事前キャッシュ** (`cache_text_encoder_outputs.py`)
  - テキストプロンプトをエンコードし、出力を保存
  - 学習時の計算負荷を軽減

### 2. LoRA学習
- **HunyuanVideo向け** (`hv_train_network.py`)
- **Wan 2.1向け** (`wan_train_network.py`)
- **FramePack向け** (`fpack_train_network.py`)

特徴：
- fp8量子化による省メモリ学習
- ブロックスワッピングによるCPUオフロード対応
- gradient checkpointing対応
- 学習中のサンプル生成機能
- TensorBoard形式のログ出力

### 3. 動画生成（推論）
- **HunyuanVideo** (`hv_generate_video.py`)
- **Wan 2.1** (`wan_generate_video.py`)
- **FramePack** (`fpack_generate_video.py`)

機能：
- LoRAの適用
- 複数の推論モード（T2V、I2V、V2V）
- バッチ生成対応
- インタラクティブモード
- SageAttentionによる高速化対応

### 4. LoRA管理
- **形式変換** (`convert_lora.py`)
  - ComfyUI形式との相互変換が可能
  - 異なるフレームワーク間でのLoRA共有を実現

- **LoRAマージ** (`merge_lora.py`)
  - LoRAを元のモデルにマージ
  - 複数のLoRAを異なる倍率でマージ可能

- **Post Hoc EMA** (`lora_post_hoc_ema.py`)
  - 学習後のモデル精度向上
  - Power Function EMAサポート

## ハードウェア要件

### 最小要件
- VRAM: 12GB以上（静止画学習）
- VRAM: 24GB以上（動画学習）
- メインメモリ: 32GB以上

### 推奨要件
- VRAM: 24GB以上
- メインメモリ: 64GB以上

### 省メモリオプション
- `--fp8_base`：fp8量子化
- `--blocks_to_swap`：CPUへのオフロード
- `--gradient_checkpointing`：メモリ使用量削減

## 使用フロー

1. **環境構築**
   - Python 3.10以上
   - PyTorch 2.5.1以上
   - 必要な依存関係のインストール

2. **モデルのダウンロード**
   - HunyuanVideo公式モデル
   - またはComfyUI提供のモデル

3. **データセット準備**
   - TOML形式の設定ファイル作成
   - 画像/動画とキャプションの準備

4. **前処理**
   - latentキャッシュの作成
   - Text Encoder出力のキャッシュ

5. **学習**
   - Accelerate設定
   - LoRA学習の実行

6. **推論・活用**
   - 学習したLoRAを使用した動画生成
   - 形式変換による他ツールでの利用

## 特徴的な機能

### 1. 省メモリ特化
- fp8量子化
- ブロックスワッピング
- VAEタイリング
- キャッシュシステム

### 2. 柔軟な学習設定
- timestep samplingの制御
- discrete flow shiftの調整
- 様々な最適化アルゴリズム

### 3. 高度な推論機能
- インタラクティブモード
- バッチ処理
- セクション単位のプロンプト制御
- SkyReels V1サポート

### 4. 開発者向け機能
- GitHub Discussionsでのコミュニティサポート
- 詳細なドキュメント
- デバッグモード
- PyTorch Dynamoによる最適化

## 開発状況

- 活発に開発中のプロジェクト
- 定期的な機能追加とバグ修正
- コミュニティからのフィードバックを歓迎
- 非公式実装（公式リポジトリとは独立）

## ライセンス

- 主要コード：Apache License 2.0
- HunyuanVideoモデル部分：オリジナルのライセンスに従う
- Wan/FramePack部分：Apache License 2.0

## 参考リンク

- [GitHub リポジトリ](https://github.com/kohya-ss/musubi-tuner)
- [ディスカッション](https://github.com/kohya-ss/musubi-tuner/discussions)
- [リリース](https://github.com/kohya-ss/musubi-tuner/releases)