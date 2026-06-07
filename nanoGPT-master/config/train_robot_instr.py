# train a miniature robot-instruction model (NL → robot commands)
# good for debugging and playing on macbooks and such

out_dir = 'out-robot-instr'
eval_interval = 100  # small dataset, evaluate frequently
eval_iters = 50
log_interval = 5

# only save when val improves (small dataset, will overfit)
always_save_checkpoint = False

wandb_log = False
wandb_project = 'robot-instr'
wandb_run_name = 'mini-gpt-robot'

dataset = 'robot_instr'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256  # instructions ~100-200 chars, JSON ~100-200 chars

# baby GPT model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.2

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 50

# on macbook also add
# device = 'cpu'
# compile = False
