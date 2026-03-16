
# DreamBooth Training Config — Session 7
TRAINING_ARGS = {
    'pretrained_model_name_or_path' : 'runwayml/stable-diffusion-v1-5',
    'instance_data_dir'  : '.../task1/dataset/images',
    'instance_prompt'    : 'a photo of sks flower',
    'output_dir'         : '.../task1/output_model',
    'resolution'         : 512,
    'train_batch_size'   : 1,
    'max_train_steps'    : 800,
    'learning_rate'      : 2e-6,
    'mixed_precision'    : 'fp16',       # saves GPU memory
    'gradient_checkpointing' : True,     # saves GPU memory
    'use_8bit_adam'      : True,         # saves GPU memory
    'checkpointing_steps': 400,
}
