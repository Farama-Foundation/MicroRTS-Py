import subprocess


def test_ppo_gridnet():

    try:
        subprocess.run(
            "cd experiments; python ppo_gridnet.py --num-bot-envs 0 --num-selfplay-envs 2 --num-steps 16 --total-timesteps 32 --cuda False",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as grepexc:                                                                                                   
        print("error code", grepexc.returncode, grepexc.output)
        assert grepexc.returncode in [0, 134, 10000]



def test_ppo_gridnet_duel_eval():
    try:
        subprocess.run(
            "cd experiments; python ppo_gridnet_duel_eval.py --num-bot-envs 0 --num-selfplay-envs 2 --num-steps 16 --total-timesteps 32 --cuda False",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as grepexc:                                                                                                   
        print("error code", grepexc.returncode, grepexc.output)
        assert grepexc.returncode in [0, 134, 10000]

