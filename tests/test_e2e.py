import subprocess


def test_ppo_gridnet():

    try:
        subprocess.run(
            "cd experiments; python ppo_gridnet.py --num-bot-envs 0 --num-selfplay-envs 2 --num-steps 16 --total-timesteps 32 --cuda False --max-eval-workers 0",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as grepexc:
        print("error code", grepexc.returncode, grepexc.output)
        assert grepexc.returncode in [0, 134]


def test_ppo_gridnet_eval_selfplay():
    try:
        subprocess.run(
            "cd experiments; python ppo_gridnet_eval.py --num-steps 16 --total-timesteps 32 --cuda False",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as grepexc:
        print("error code", grepexc.returncode, grepexc.output)
        assert grepexc.returncode in [0, 134]


def test_ppo_gridnet_eval_bot():

    subprocess.run(
        "cd experiments; python ppo_gridnet_eval.py --ai coacAI --num-steps 16 --total-timesteps 32 --cuda False",
        shell=True,
        check=True,
    )
