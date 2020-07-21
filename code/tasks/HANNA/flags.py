import argparse

def make_parser():
   parser = argparse.ArgumentParser()

   # Meta
   parser.add_argument('-config_file', type=str,
        help='configuration file')
   parser.add_argument('-data_dir', type=str, default='hanna',
        help='data directory')
   parser.add_argument('-data_prefix', type=str, default='hanna',
        help='data file prefix')
   parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
   parser.add_argument('-exp_name', type=str,
        help='name of the experiment')
   parser.add_argument('-seed', type=int,
        help='random seed')
   parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
   parser.add_argument('-max_instr_len', type=int,
        help='maximum number of words in a language instruction')
   parser.add_argument('-device_id', type=int,
        help='gpu id')
   parser.add_argument('-start_point_radius', type=int,
        help='Start point radius')

   # Model
   parser.add_argument('-img_feature_size', type=int, default=2048,
        help='image embedding size')
   parser.add_argument('-word_embed_size', type=int,
        help='word embedding size')
   parser.add_argument('-action_embed_size', type=int,
        help='navigation action embedding size')
   parser.add_argument('-ask_embed_size', type=int,
        help='ask action embedding size')
   parser.add_argument('-loc_embed_size', type=int,
        help='camera angles embedding size')
   parser.add_argument('-hidden_size', type=int,
        help='number of hidden units')
   parser.add_argument('-attention_heads', type=int,
        help='Number of attention heads')
   parser.add_argument('-num_layers', type=int,
        help='Number of transformer layers')
   parser.add_argument('-dropout_ratio', type=float,
        help='dropout probability')

   # Training
   parser.add_argument('-alpha', type=float,
        help='Curiosity-encouraging loss weight')
   parser.add_argument('-lr', type=float,
        help='learning rate')
   parser.add_argument('-weight_decay', type=float,
        help='L2-regularization weight')
   parser.add_argument('-n_iters', type=int,
        help='number of training iterations (batches)')
   parser.add_argument('-batch_size', type=int,
        help='batch size (both training and evaluation)')
   parser.add_argument('-train_episode_len', type=int,
        help='maximum number of time steps during training')
   parser.add_argument('-start_lr_decay', type=int,
        help='iteration to start decaying learning rate')
   parser.add_argument('-lr_decay_rate', type=float,
        help='learning rate decay rate')
   parser.add_argument('-decay_lr_every', type=int,
        help='number of iterations between learning rate decays')
   parser.add_argument('-log_every', type=int,
        help='number of iterations between information loggings')

   # Evaluation
   parser.add_argument('-eval_episode_len', type=int,
        help='maximum number of time steps during evaluation')
   parser.add_argument('-success_radius', type=float,
        help='success radius')
   parser.add_argument('-eval_only', type=int,
        help='evaluation mode')
   parser.add_argument('-eval_on_val', type=int, default=0)


   # Non-learning baselines
   parser.add_argument('-random_agent', type=int, default=0,
        help='Agent that randomly selects navigation actions')
   parser.add_argument('-forward_agent', type=int, default=0,
        help='Agent that always selects action 1')
   parser.add_argument('-shortest_agent', type=int, default=0,
        help='Optimal shortest-path agent')

   # Perfect language interpretation baseline
   parser.add_argument('-perfect_interpretation', type=int, default=0,
        help='provide perfect assistance interpretation')

   # Ask baseline
   parser.add_argument('-ask_baseline', type=str, default=None,
        help='Help-request teacher baseline')

   # Ablation baseline
   parser.add_argument('-no_sim_attend', type=int, default=0,
        help="No cosine similarity attention (beta = 0)")
   parser.add_argument('-no_reason', type=int, default=0,
        help="No condition (reason) prediction")
   parser.add_argument('-no_reset_inter', type=int, default=0,
        help="No reset inter-task module")

   # Single modality baseline
   parser.add_argument('-instruction_baseline', type=str, default=None,
        help='Instruction type baseline')

   # for dicencoder
   parser.add_argument('--encoder_type', dest='encoder_type', default='dic', type=str, help='encoder type')
   parser.add_argument('--d_hidden_size', dest='d_hidden_size', default=256, type=int, help='decoder hidden_size')
   parser.add_argument('--d_ctx_size', dest='d_ctx_size', default=768, type=int, help='ctx hidden_size')
   parser.add_argument('--d_enc_hidden_size', dest='d_enc_hidden_size', default=128, type=int, help='encoder hidden_size')
   parser.add_argument('--d_dropout_ratio', dest='d_dropout_ratio', default=0.4, type=float, help='dropout_ratio')
   parser.add_argument('--d_bidirectional', dest='d_bidirectional', type=bool, default=True, help='bidirectional')
   parser.add_argument('--d_transformer_update', dest='d_transformer_update', type=bool, default=False, help='update Bert')
   parser.add_argument('--d_update_add_layer', dest='d_update_add_layer', type=bool, default=False, help='update fusion layer in Bert')
   parser.add_argument('--d_bert_n_layers', dest='d_bert_n_layers', type=int, default=1, help='bert_n_layers')
   parser.add_argument('--d_reverse_input', dest='d_reverse_input', type=bool, default=True, help='reverse')
   parser.add_argument('--d_top_lstm', dest='d_top_lstm', type=bool, default=True, help='add lstm to the top of transformers')
   parser.add_argument('--d_vl_layers', dest='d_vl_layers', type=int, default=4, help='vl_layers')
   parser.add_argument('--d_la_layers', dest='d_la_layers', type=int, default=9, help='la_layers')
   parser.add_argument('--d_bert_type', dest='d_bert_type', type=str, default="small", help='small or large')
   parser.add_argument('--pretrain_model_name', dest='pretrain_model_name', type=str, default=None, help='the name of pretrained model')

   return parser
