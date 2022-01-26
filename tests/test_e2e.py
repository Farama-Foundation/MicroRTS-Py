import subprocess


def test_ppo_gridnet():
    subprocess.run(
        "cd experiments; python ppo_gridnet.py --num-bot-envs 0 --num-selfplay-envs 2 --num-steps 16 --total-timesteps 32 --cuda False",
        shell=True,
        check=True,
    )


def test_ppo_gridnet_duel_eval():
    subprocess.run(
        "cd experiments; python ppo_gridnet_duel_eval.py --num-bot-envs 0 --num-selfplay-envs 2 --num-steps 16 --total-timesteps 32 --cuda False",
        shell=True,
        check=True,
    )
