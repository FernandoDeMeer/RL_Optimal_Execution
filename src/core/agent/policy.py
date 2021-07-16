# from ray.rllib.policy.tf_policy_template import build_tf_policy
# from ray.rllib.utils.explained_variance import explained_variance
# from ray.rllib.utils.tf_ops import make_tf_callable
# from ray.rllib.utils import try_import_tf
#
# tf = try_import_tf()
#
#
#
# PPOTFPolicy = build_tf_policy(
#     name="PPOTFPolicy",
#     get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
#     loss_fn=ppo_surrogate_loss,
#     stats_fn=kl_and_loss_stats,
#     extra_action_fetches_fn=vf_preds_and_logits_fetches,
#     postprocess_fn=postprocess_ppo_gae,
#     gradients_fn=clip_gradients,
#     before_init=setup_config,
#     before_loss_init=setup_mixins,
#     mixins=[
#         LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
#         ValueNetworkMixin
#     ])
