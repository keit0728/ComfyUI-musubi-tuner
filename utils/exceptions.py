"""
MusubiTuner共通の例外クラス
"""


class MusubiTunerError(Exception):
    """MusubiTuner関連のエラーを表すカスタム例外クラス"""

    def __init__(self, message, stderr=None, returncode=None):
        """
        Args:
            message (str): エラーメッセージ
            stderr (str, optional): 標準エラー出力
            returncode (int, optional): プロセスの終了コード
        """
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode