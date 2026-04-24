import argparse
import os

from qpt.executor import CreateExecutableModule as CEM
from qpt.kernel.qinterpreter import (
    DISPLAY_LOCAL_INSTALL,
    DISPLAY_ONLINE_INSTALL,
    DISPLAY_SETUP_INSTALL,
    PYPI_PIP_SOURCE,
)
from qpt.modules.package import CustomPackage
from qpt.smart_opt import set_default_pip_source


DEPLOY_MODE_MAP = {
    "setup_install": DISPLAY_SETUP_INSTALL,
    "local_install": DISPLAY_LOCAL_INSTALL,
    "online_install": DISPLAY_ONLINE_INSTALL,
}

GENERATED_DIRS = [
    "installer-dist",
    "installer-dist-cu118-ascii",
    "installer-dist-cu126-ascii",
    "vsr_out",
    "vsr_out_cu118",
    "vsr_out_cu126",
]


def main():
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    launch_path = os.path.join(work_dir, "gui.py")
    save_path = os.environ.get("VSR_OUT_DIR") or os.path.join(os.path.dirname(work_dir), "vsr_out")
    icon_path = os.path.join(work_dir, "design", "vsr.ico")
    ignore_dirs = [os.path.join(work_dir, dirname) for dirname in GENERATED_DIRS]

    parser = argparse.ArgumentParser(description="Build the VSR distributable bundle.")
    parser.add_argument(
        "--cuda",
        nargs="?",
        const="11.8",
        default=None,
        help="Include CUDA runtime support. Examples: --cuda, --cuda 11.8, --cuda 12.6",
    )
    parser.add_argument(
        "--directml",
        nargs="?",
        const=True,
        default=None,
        help="Include DirectML runtime support.",
    )
    parser.add_argument(
        "--deploy-mode",
        choices=tuple(DEPLOY_MODE_MAP.keys()),
        default=os.environ.get("VSR_DEPLOY_MODE", "setup_install"),
        help=(
            "QPT dependency deployment mode. "
            "Default is setup_install so dependencies are preinstalled into Release/"
            "Python and first-run offline installation is avoided."
        ),
    )
    args = parser.parse_args()

    deploy_mode = DEPLOY_MODE_MAP[args.deploy_mode]
    sub_modules = []

    if args.cuda == "11.8":
        sub_modules.append(
            CustomPackage(
                "torch==2.7.0 torchvision==0.22.0",
                deploy_mode=deploy_mode,
                find_links=PYPI_PIP_SOURCE,
                opts="--index-url https://download.pytorch.org/whl/cu118 ",
            )
        )
    elif args.cuda == "12.6":
        sub_modules.append(
            CustomPackage(
                "torch==2.7.0 torchvision==0.22.0",
                deploy_mode=deploy_mode,
                find_links=PYPI_PIP_SOURCE,
                opts="--index-url https://download.pytorch.org/whl/cu126 ",
            )
        )
    elif args.cuda == "12.8":
        sub_modules.append(
            CustomPackage(
                "torch==2.7.0 torchvision==0.22.0",
                deploy_mode=deploy_mode,
                find_links=PYPI_PIP_SOURCE,
                opts="--index-url https://download.pytorch.org/whl/cu128 ",
            )
        )

    if args.directml:
        sub_modules.append(CustomPackage("torch_directml==0.2.5.dev240914", deploy_mode=deploy_mode))

    if os.getenv("QPT_Action") == "True":
        set_default_pip_source(PYPI_PIP_SOURCE)

    module = CEM(
        work_dir=work_dir,
        launcher_py_path=launch_path,
        save_path=save_path,
        ignore_dirs=ignore_dirs,
        icon=icon_path,
        hidden_terminal=False,
        requirements_file="./requirements.txt",
        deploy_mode=deploy_mode,
        sub_modules=sub_modules,
    )
    module.make()


if __name__ == "__main__":
    main()
