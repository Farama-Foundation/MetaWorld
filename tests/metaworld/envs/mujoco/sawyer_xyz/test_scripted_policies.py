import pytest

from metaworld import MT1
from metaworld.policies import (
    SawyerAssemblyV3Policy,
    SawyerBasketballV3Policy,
    SawyerBinPickingV3Policy,
    SawyerBoxCloseV3Policy,
    SawyerButtonPressTopdownV3Policy,
    SawyerButtonPressTopdownWallV3Policy,
    SawyerButtonPressV3Policy,
    SawyerButtonPressWallV3Policy,
    SawyerCoffeeButtonV3Policy,
    SawyerCoffeePullV3Policy,
    SawyerCoffeePushV3Policy,
    SawyerDialTurnV3Policy,
    SawyerDisassembleV3Policy,
    SawyerDoorCloseV3Policy,
    SawyerDoorLockV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDoorUnlockV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerFaucetCloseV3Policy,
    SawyerFaucetOpenV3Policy,
    SawyerHammerV3Policy,
    SawyerHandInsertV3Policy,
    SawyerHandlePressSideV3Policy,
    SawyerHandlePressV3Policy,
    SawyerHandlePullSideV3Policy,
    SawyerHandlePullV3Policy,
    SawyerLeverPullV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerPegUnplugSideV3Policy,
    SawyerPickOutOfHoleV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerPickPlaceWallV3Policy,
    SawyerPlateSlideBackSideV3Policy,
    SawyerPlateSlideBackV3Policy,
    SawyerPlateSlideSideV3Policy,
    SawyerPlateSlideV3Policy,
    SawyerPushBackV3Policy,
    SawyerPushV3Policy,
    SawyerPushWallV3Policy,
    SawyerReachV3Policy,
    SawyerReachWallV3Policy,
    SawyerShelfPlaceV3Policy,
    SawyerSoccerV3Policy,
    SawyerStickPullV3Policy,
    SawyerStickPushV3Policy,
    SawyerSweepIntoV3Policy,
    SawyerSweepV3Policy,
    SawyerWindowCloseV3Policy,
    SawyerWindowOpenV3Policy,
)

policies = dict(
    {
        "assembly-v3": SawyerAssemblyV3Policy,
        "basketball-v3": SawyerBasketballV3Policy,
        "bin-picking-v3": SawyerBinPickingV3Policy,
        "box-close-v3": SawyerBoxCloseV3Policy,
        "button-press-topdown-v3": SawyerButtonPressTopdownV3Policy,
        "button-press-topdown-wall-v3": SawyerButtonPressTopdownWallV3Policy,
        "button-press-v3": SawyerButtonPressV3Policy,
        "button-press-wall-v3": SawyerButtonPressWallV3Policy,
        "coffee-button-v3": SawyerCoffeeButtonV3Policy,
        "coffee-pull-v3": SawyerCoffeePullV3Policy,
        "coffee-push-v3": SawyerCoffeePushV3Policy,
        "dial-turn-v3": SawyerDialTurnV3Policy,
        "disassemble-v3": SawyerDisassembleV3Policy,
        "door-close-v3": SawyerDoorCloseV3Policy,
        "door-lock-v3": SawyerDoorLockV3Policy,
        "door-open-v3": SawyerDoorOpenV3Policy,
        "door-unlock-v3": SawyerDoorUnlockV3Policy,
        "drawer-close-v3": SawyerDrawerCloseV3Policy,
        "drawer-open-v3": SawyerDrawerOpenV3Policy,
        "faucet-close-v3": SawyerFaucetCloseV3Policy,
        "faucet-open-v3": SawyerFaucetOpenV3Policy,
        "hammer-v3": SawyerHammerV3Policy,
        "hand-insert-v3": SawyerHandInsertV3Policy,
        "handle-press-side-v3": SawyerHandlePressSideV3Policy,
        "handle-press-v3": SawyerHandlePressV3Policy,
        "handle-pull-v3": SawyerHandlePullV3Policy,
        "handle-pull-side-v3": SawyerHandlePullSideV3Policy,
        "peg-insert-side-v3": SawyerPegInsertionSideV3Policy,
        "lever-pull-v3": SawyerLeverPullV3Policy,
        "peg-unplug-side-v3": SawyerPegUnplugSideV3Policy,
        "pick-out-of-hole-v3": SawyerPickOutOfHoleV3Policy,
        "pick-place-v3": SawyerPickPlaceV3Policy,
        "pick-place-wall-v3": SawyerPickPlaceWallV3Policy,
        "plate-slide-back-side-v3": SawyerPlateSlideBackSideV3Policy,
        "plate-slide-back-v3": SawyerPlateSlideBackV3Policy,
        "plate-slide-side-v3": SawyerPlateSlideSideV3Policy,
        "plate-slide-v3": SawyerPlateSlideV3Policy,
        "reach-v3": SawyerReachV3Policy,
        "reach-wall-v3": SawyerReachWallV3Policy,
        "push-back-v3": SawyerPushBackV3Policy,
        "push-v3": SawyerPushV3Policy,
        "push-wall-v3": SawyerPushWallV3Policy,
        "shelf-place-v3": SawyerShelfPlaceV3Policy,
        "soccer-v3": SawyerSoccerV3Policy,
        "stick-pull-v3": SawyerStickPullV3Policy,
        "stick-push-v3": SawyerStickPushV3Policy,
        "sweep-into-v3": SawyerSweepIntoV3Policy,
        "sweep-v3": SawyerSweepV3Policy,
        "window-close-v3": SawyerWindowCloseV3Policy,
        "window-open-v3": SawyerWindowOpenV3Policy,
    }
)


@pytest.mark.parametrize("env_name", MT1.ENV_NAMES)
def test_policy(env_name):
    mt1 = MT1(env_name)
    env = mt1.train_classes[env_name]()
    p = policies[env_name]()
    completed = 0
    for task in mt1.train_tasks:
        env.set_task(task)
        obs, info = env.reset()
        done = False
        count = 0
        while count < 500 and not done:
            count += 1
            a = p.get_action(obs)
            next_obs, _, trunc, termn, info = env.step(a)
            done = trunc or termn
            obs = next_obs
            if int(info["success"]) == 1:
                completed += 1
                break
    assert (float(completed) / 50) >= 0.80
