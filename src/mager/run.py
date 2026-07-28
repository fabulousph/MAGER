"""Command line entry point for MAGER."""
import argparse

from config import set_dataset, DATASETS
from detector import FewShotSelectedMetaPathDetector, _load_meta_paths_from_json
from evolve import evolve_and_evaluate
from meta_path import MetaPath, MetaPathInitializer
from preprocess import enhance_dataset


DEFAULT_METAPATHS = [
    MetaPath(path=[("User", "Comment"), ("News", "")], description="用户评论新闻"),
    MetaPath(path=[("News", "Repost"), ("User", "")], description="新闻被转发"),
    MetaPath(path=[("User", "Reply"), ("User", "Repost"), ("News", "")], description="用户回复用户后转发新闻"),
]


def load_meta_paths(args):
    if args.meta_path_json:
        return _load_meta_paths_from_json(args.meta_path_json)

    if args.meta_path_indices:
        initial = MetaPathInitializer.get_initial_meta_paths()
        selected = [initial[i] for i in args.meta_path_indices if 0 <= i < len(initial)]
        if len(selected) < 2:
            raise ValueError("At least two meta-paths are required.")
        return selected[:3]

    return DEFAULT_METAPATHS


def evaluate(args):
    set_dataset(args.dataset)
    meta_paths = load_meta_paths(args)
    detector = FewShotSelectedMetaPathDetector(
        model_config_name=args.model_config,
        selected_meta_paths=meta_paths,
        icl_pool_size=args.icl_pool_size,
        top_k_examples=args.top_k,
    )
    return detector.evaluate(
        limit=args.limit,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )


def evolve(args):
    set_dataset(args.dataset)
    return evolve_and_evaluate(
        dataset_limit=args.evolve_limit,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        use_enhanced_dataset=not args.raw_dataset,
        test_limit=args.test_limit,
    )


def preprocess(args):
    set_dataset(args.dataset)
    enhance_dataset(limit=args.limit)


def list_meta_paths():
    for i, meta_path in enumerate(MetaPathInitializer.get_initial_meta_paths()):
        print(f"[{i}] {meta_path.description}: {meta_path.to_formal_string()}")


def main():
    parser = argparse.ArgumentParser(description="MAGER: evolutionary meta-path reasoning for fake news detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="politifact")
    common.add_argument("--model_config", default="agent_d")

    eval_parser = subparsers.add_parser("evaluate", parents=[common], help="evaluate with selected/evolved meta-paths")
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--output_dir", default="results")
    eval_parser.add_argument("--no_save", action="store_true")
    eval_parser.add_argument("--icl_pool_size", type=int, default=100)
    eval_parser.add_argument("--top_k", type=int, default=4)
    eval_parser.add_argument("--meta_path_json", type=str, default=None)
    eval_parser.add_argument("--meta_path_indices", type=int, nargs="+")
    eval_parser.set_defaults(func=evaluate)

    evolve_parser = subparsers.add_parser("evolve", parents=[common], help="evolve meta-paths and evaluate them")
    evolve_parser.add_argument("--evolve_limit", type=int, default=100)
    evolve_parser.add_argument("--test_limit", type=int, default=200)
    evolve_parser.add_argument("--max_iterations", type=int, default=100)
    evolve_parser.add_argument("--population_size", type=int, default=20)
    evolve_parser.add_argument("--raw_dataset", action="store_true")
    evolve_parser.set_defaults(func=evolve)

    preprocess_parser = subparsers.add_parser("preprocess", parents=[common], help="generate enhanced dataset with the enhancement agent")
    preprocess_parser.add_argument("--limit", type=int, default=400, help="number of fake and real news items to process separately")
    preprocess_parser.set_defaults(func=preprocess)

    list_parser = subparsers.add_parser("list-metapaths", help="list initial meta-path candidates")
    list_parser.set_defaults(func=lambda args: list_meta_paths())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
