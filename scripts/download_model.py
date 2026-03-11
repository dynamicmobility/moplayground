import argparse
import minimal_mjx as mm
from pathlib import Path

# Dictionary mapping environment names to wandb run IDs
ENV_RUN_IDS = {
    'cheetah':       '4303yymq',
    'hopper':        '0000000',
    'walker':        '0000000',
    'ant':           '0000000',
    'humanoid':      '0000000',
    'BRUCE':         '0000000',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Download a pre-trained policy from wandb')
    parser.add_argument('--save_dir', type=str, default='results/wandb-downloads',
                        help='Directory to save the downloaded model (default: wandb-downloads)')
    parser.add_argument('--env', type=str, required=True,
                        choices=list(ENV_RUN_IDS.keys()),
                        help='Environment name to download the policy for')
    return parser.parse_args()

def main():
    args = parse_args()
    
    env_name = args.env
    run_id = ENV_RUN_IDS[env_name]
    save_dir = Path(args.save_dir)
    
    print(f"Downloading policy for {env_name} (run_id: {run_id})...")
    print(f"Saving to: {save_dir / env_name}")
    
    mm.utils.logging.download_model(
        run_id        = run_id,
        save_dir      = save_dir,
        model_name    = env_name,
        entity        = 'njanwani-gatech',
        project       = 'MO-Playground-Official'
    )
    
    print(f"Done! Config saved to: {save_dir / env_name / 'config.yaml'}")

if __name__ == '__main__':
    main()
