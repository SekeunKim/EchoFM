from main_pretrain import get_args_parser, main
from pathlib import Path

def invoke_main() -> None:
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

if __name__ == "__main__":
    invoke_main() 
