# Motivation

This is to help with reformatting our trl fork to be easier to work with.

What are the current functions, what arguments do they take, and what do they do?



| Function | Description | List of Arguments |
| -- | -- | -- |
| batch_update | tbc | tbc |
| rollouts_to_loss_variables | Takes in an iteration batch of queries and responses, and computes various variables needed for defining the loss function. | queries, query_responses, logitss, ref_policy, unwrapped_value_model, reward_model, processing_class, reward_model_processing_class, context_length, stop_token_id, response_truncation_sequences: list[list[int]], local_rollout_forward_batch_size, ref_temperature, device |
| batch_generation | tbc | tbc |
| micro_batch_updates | tbc | tbc |
| forward_pass_on_rollouts | tbc | tbc |
| prioritise_batch | tbc | tbc |