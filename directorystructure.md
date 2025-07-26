# ディレクトリ構成

以下のディレクトリ構造に従って実装を行ってください：

```
/
├── .cursor/                             # Cursorエディタ設定
├── nodes/                               # ComfyUIカスタムノード実装
├── example_workflows/                   # サンプルワークフロー・使用例
├── .cursorignore                        # Cursor除外設定
├── .gitignore                           # Git除外設定
├── __init__.py                          # プロジェクトパッケージ初期化ファイル
├── requirements.txt                     # Python依存関係定義
├── coding-rule.md                       # コーディング規約
├── directorystructure.md                # ディレクトリ構成定義（このファイル）
├── README.md                            # プロジェクト説明
└── technologystack.md                   # 技術スタック定義
```

## 開発時の注意点

- 新しいノードを追加する場合は、`nodes/`ディレクトリ内に適切な名前でPythonファイルを作成してください
- 各ノードファイルには適切なクラス定義とComfyUI用の登録処理を含めてください
- `__init__.py`ファイルを適切に更新して、新しいノードが正しく認識されるようにしてください
- サンプルワークフローや使用例は`example_workflows/`ディレクトリに配置してください
- 新しい依存関係を追加する場合は、`requirements.txt`ファイルを更新してください
