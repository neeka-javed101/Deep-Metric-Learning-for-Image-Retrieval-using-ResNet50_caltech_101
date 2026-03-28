from argparse import Namespace
from train import run_single_experiment
def main():
    experiments = [
        ("resnet50_contrastive", "contrastive"),
        ("resnet50_triplet", "triplet"),
        ("resnet50_batch_hard_triplet", "batch_hard_triplet"),
    ]

    for model_name, loss_type in experiments:
        print(f"\n🚀 Starting experiment: {model_name} with {loss_type} loss")
        args = Namespace(
            model_name=model_name,
            embedding_dimension=128,
            loss_type=loss_type,
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-4,
            margin=1.0,
            num_workers=4,
        )
        run_single_experiment(args)