"""可复验基线；ASCENT-derived为官方逐MBS阈值gate的动态多用户扩展。"""


def no_coordination_action(env):
    return env.reference_action()


def ascent_action(env):
    """每MBS贡献对任一NTN超阈则静默，下一快照重新评估，可恢复。"""
    return env.baseline_action('ASCENT-derived')
