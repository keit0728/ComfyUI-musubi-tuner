"""
Wan2.1動画生成のためのコマンドビルダーノード
"""

import json
import os
import sys
import subprocess
import threading
import queue
from pathlib import Path

from ..utils.exceptions import MusubiTunerError

# タスクタイプの定義
TASK_CHOICES = [
    "t2v-1.3B",  # Text to Video 1.3B
    "t2v-14B",  # Text to Video 14B
    "i2v-14B",  # Image to Video 14B
    "t2i-14B",  # Text to Image 14B
    "flf2v-14B",  # First/Last Frame to Video 14B
    "t2v-1.3B-FC",  # Fun Control T2V 1.3B
    "t2v-14B-FC",  # Fun Control T2V 14B
    "i2v-14B-FC",  # Fun Control I2V 14B
]

# アテンションモード
ATTN_MODES = ["torch", "sdpa", "xformers", "sageattn", "flash", "flash2", "flash3"]

# CFGスキップモード
CFG_SKIP_MODES = ["none", "early", "late", "middle", "early_late", "alternate"]

# 出力タイプ
OUTPUT_TYPES = ["video", "images", "latent", "both", "latent_images"]

# サポートされるビデオサイズ
SUPPORTED_SIZES = {
    "t2v-14B": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
    "t2v-1.3B": [(480, 832), (832, 480)],
    "i2v-14B": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
    "t2i-14B": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
    "flf2v-14B": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
    "t2v-1.3B-FC": [(480, 832), (832, 480)],
    "t2v-14B-FC": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
    "i2v-14B-FC": [(720, 1280), (1280, 720), (480, 832), (832, 480)],
}


class MusubiTunerWanGenerateVideo:
    """Wan2.1動画生成ノード"""

    def __init__(self):
        self.musubi_tuner_path = None
        self.venv_path = None
        self.python_executable = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (TASK_CHOICES, {"default": "t2v-14B"}),
                "prompt": ("STRING", {"multiline": True}),
                "dit": ("STRING", {"default": ""}),
                "vae": ("STRING", {"default": ""}),
                "t5": ("STRING", {"default": ""}),
                "video_size_width": (
                    "INT",
                    {"default": 832, "min": 256, "max": 2048, "step": 16},
                ),
                "video_size_height": (
                    "INT",
                    {"default": 480, "min": 256, "max": 2048, "step": 16},
                ),
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
                "musubi_tuner_path": ("STRING", {"default": ""}),
                "working_directory": ("STRING", {"default": ""}),
                "environment_json": ("STRING", {"default": "{}", "multiline": True}),
                "timeout": ("INT", {"default": 3600, "min": 1, "max": 86400}),
                "show_progress": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("stdout", "stderr", "return_code")
    FUNCTION = "execute_command"
    CATEGORY = "MusubiTuner/Wan2.1"

    def validate_inputs(self, **kwargs):
        """入力パラメータの検証"""

        # タスクとビデオサイズの整合性チェック
        task = kwargs["task"]
        video_size = (kwargs["video_size_width"], kwargs["video_size_height"])

        if task in SUPPORTED_SIZES:
            if video_size not in SUPPORTED_SIZES[task]:
                supported_str = ", ".join(
                    [f"{w}x{h}" for w, h in SUPPORTED_SIZES[task]]
                )
                raise ValueError(
                    f"Invalid video size {video_size[0]}x{video_size[1]} for task {task}. "
                    f"Supported sizes: {supported_str}"
                )

        # I2Vタスクの場合はCLIPパスが必須
        if "i2v" in task and not kwargs.get("clip"):
            raise ValueError("CLIP model path is required for I2V tasks")

        # Fun Controlタスクの場合は制御パスが必須
        if "FC" in task and not kwargs.get("control_path"):
            raise ValueError("Control path is required for Fun Control tasks")

    def _validate_musubi_tuner_path(self, path_str):
        """musubi-tunerのパスを検証し、必要な情報を設定"""
        if not path_str:
            raise MusubiTunerError("musubi-tunerのパスを指定してください。")

        path = Path(path_str)
        if not path.exists():
            raise MusubiTunerError(f"指定されたパスが存在しません: {path_str}")

        if not (path / "src" / "musubi_tuner").exists():
            raise MusubiTunerError(
                f"指定されたパスにsrc/musubi_tunerディレクトリが見つかりません: {path_str}\n"
                "正しいmusubi-tunerのインストールディレクトリを指定してください。"
            )

        self.musubi_tuner_path = path

        # 仮想環境を探す
        venv_candidates = [
            path / "venv",
            path / ".venv",
            path / "env",
        ]

        for venv_path in venv_candidates:
            if venv_path.exists():
                self.venv_path = venv_path
                break
        else:
            raise MusubiTunerError(
                f"musubi-tunerの仮想環境が見つかりません: {path}\n"
                "musubi-tunerが正しくセットアップされているか確認してください。"
            )

        # Python実行ファイルのパスを取得
        if sys.platform == "win32":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"

        if not python_exe.exists():
            raise MusubiTunerError(
                f"Python実行可能ファイルが見つかりません: {python_exe}"
            )

        self.python_executable = str(python_exe)

    def _execute_command(
        self,
        cmd_args,
        env_dict=None,
        progress_callback=None,
        timeout=None,
        working_directory=None,
    ):
        """
        コマンドを実行し、出力を返す

        Args:
            cmd_args (list): コマンドライン引数のリスト
            env_dict (dict, optional): 追加の環境変数
            progress_callback (callable, optional): 進捗コールバック関数
            timeout (float, optional): タイムアウト秒数
            working_directory (str, optional): 作業ディレクトリ

        Returns:
            str: コマンドの標準出力

        Raises:
            MusubiTunerError: 実行エラー時
        """
        # 完全なコマンドを構築
        full_cmd = [self.python_executable] + cmd_args

        # 作業ディレクトリを設定（指定がなければmusubi_tuner_path/src/musubi_tunerを使用）
        cwd = (
            working_directory
            if working_directory
            else str(self.musubi_tuner_path / "src" / "musubi_tuner")
        )

        # 環境変数を設定
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.musubi_tuner_path / "src")

        # 追加の環境変数を適用
        if env_dict:
            env.update(env_dict)

        # プロセスを起動
        try:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            raise MusubiTunerError(f"コマンドの起動に失敗しました: {e}")

        # 標準出力と標準エラー出力を収集
        stdout_lines = []
        stderr_lines = []

        # 出力を非同期で読み取るためのキューとスレッド
        q_stdout = queue.Queue()
        q_stderr = queue.Queue()

        def enqueue_output(file, queue):
            for line in iter(file.readline, ""):
                queue.put(line)
            file.close()

        # スレッドを開始
        t_stdout = threading.Thread(
            target=enqueue_output, args=(process.stdout, q_stdout)
        )
        t_stderr = threading.Thread(
            target=enqueue_output, args=(process.stderr, q_stderr)
        )
        t_stdout.daemon = True
        t_stderr.daemon = True
        t_stdout.start()
        t_stderr.start()

        # 出力を処理
        while True:
            # 標準出力を処理
            try:
                line = q_stdout.get_nowait()
                stdout_lines.append(line)
                if progress_callback:
                    progress_callback(line.strip())
            except queue.Empty:
                pass

            # 標準エラー出力を処理
            try:
                line = q_stderr.get_nowait()
                stderr_lines.append(line)
            except queue.Empty:
                pass

            # プロセスが終了したかチェック
            if process.poll() is not None:
                # 残りの出力を読み取る
                while not q_stdout.empty():
                    line = q_stdout.get()
                    stdout_lines.append(line)
                    if progress_callback:
                        progress_callback(line.strip())

                while not q_stderr.empty():
                    line = q_stderr.get()
                    stderr_lines.append(line)

                break

        # プロセスの終了を待つ
        returncode = process.wait()

        # 結果を処理
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        if returncode != 0:
            raise MusubiTunerError(
                f"コマンドの実行に失敗しました (終了コード: {returncode})",
                stderr=stderr,
                returncode=returncode,
            )

        return stdout

    def execute_command(self, **kwargs):
        """Wan2.1コマンドの実行"""

        # 入力検証
        try:
            self.validate_inputs(**kwargs)
        except ValueError as e:
            raise MusubiTunerError(str(e))

        # outputディレクトリのパスを自動設定
        current_dir = Path(__file__).parent.parent  # ComfyUI-musubi-tunerディレクトリ
        output_dir = current_dir / "output"
        output_dir.mkdir(exist_ok=True)  # ディレクトリが存在しない場合は作成

        cmd_args = [
            "wan_generate_video.py",
            "--task",
            kwargs["task"],
            "--prompt",
            kwargs["prompt"],
            "--save_path",
            str(output_dir),
            "--dit",
            kwargs["dit"],
            "--vae",
            kwargs["vae"],
            "--t5",
            kwargs["t5"],
            "--video_size",
            str(kwargs["video_size_width"]),
            str(kwargs["video_size_height"]),
            "--video_length",
            str(kwargs["video_length"]),
            "--fps",
            str(kwargs["fps"]),
            "--infer_steps",
            str(kwargs["infer_steps"]),
            "--attn_mode",
            kwargs["attn_mode"],
            "--output_type",
            kwargs["output_type"],
        ]

        # オプションパラメータの追加
        if kwargs["seed"] >= 0:
            cmd_args.extend(["--seed", str(kwargs["seed"])])

        if kwargs["negative_prompt"]:
            cmd_args.extend(["--negative_prompt", kwargs["negative_prompt"]])

        if kwargs["guidance_scale"] != 5.0:
            cmd_args.extend(["--guidance_scale", str(kwargs["guidance_scale"])])

        if kwargs["flow_shift"] != -1.0:
            cmd_args.extend(["--flow_shift", str(kwargs["flow_shift"])])

        if kwargs["fp8"]:
            cmd_args.append("--fp8")

        if kwargs["fp8_scaled"]:
            cmd_args.append("--fp8_scaled")

        if kwargs["fp8_fast"]:
            cmd_args.append("--fp8_fast")

        if kwargs["fp8_t5"]:
            cmd_args.append("--fp8_t5")

        if kwargs["blocks_to_swap"] > 0:
            cmd_args.extend(["--blocks_to_swap", str(kwargs["blocks_to_swap"])])

        if kwargs["vae_cache_cpu"]:
            cmd_args.append("--vae_cache_cpu")

        # I2V/V2V関連
        if kwargs["image_path"]:
            cmd_args.extend(["--image_path", kwargs["image_path"]])

        if kwargs["end_image_path"]:
            cmd_args.extend(["--end_image_path", kwargs["end_image_path"]])

        if kwargs["video_path"]:
            cmd_args.extend(["--video_path", kwargs["video_path"]])

        if kwargs["clip"]:
            cmd_args.extend(["--clip", kwargs["clip"]])

        # Fun Control関連
        if kwargs["control_path"]:
            cmd_args.extend(["--control_path", kwargs["control_path"]])

        # LoRA設定
        if kwargs["lora_weight"]:
            lora_weights = kwargs["lora_weight"].split(",")
            lora_multipliers = str(kwargs["lora_multiplier"]).split(",")

            cmd_args.append("--lora_weight")
            cmd_args.extend(lora_weights)
            cmd_args.append("--lora_multiplier")
            cmd_args.extend(lora_multipliers)

        # 高度な設定
        if kwargs["cfg_skip_mode"] != "none":
            cmd_args.extend(["--cfg_skip_mode", kwargs["cfg_skip_mode"]])

        if kwargs["cfg_apply_ratio"] != 1.0:
            cmd_args.extend(["--cfg_apply_ratio", str(kwargs["cfg_apply_ratio"])])

        if kwargs["trim_tail_frames"] > 0:
            cmd_args.extend(["--trim_tail_frames", str(kwargs["trim_tail_frames"])])

        if kwargs["cpu_noise"]:
            cmd_args.append("--cpu_noise")

        # musubi-tunerパスの検証
        self._validate_musubi_tuner_path(kwargs.get("musubi_tuner_path", ""))

        # 環境変数を解析
        try:
            env_dict = json.loads(kwargs.get("environment_json", "{}"))
        except json.JSONDecodeError as e:
            raise MusubiTunerError(f"環境変数の解析エラー: {e}")

        # 進捗コールバック
        def progress_callback(line):
            if kwargs.get("show_progress", True):
                print(f"[MusubiTuner] {line}")

        # コマンド実行
        stdout = self._execute_command(
            cmd_args,
            env_dict=env_dict,
            progress_callback=progress_callback,
            timeout=kwargs.get("timeout", 3600),
            working_directory=(
                kwargs.get("working_directory")
                if kwargs.get("working_directory")
                else None
            ),
        )

        return (stdout, "", 0)
