import os
import torch
from torch.optim import Adam
import numpy as np
import random
import argparse
import logging
from tqdm import tqdm, trange

from utils_LATTE import DataUtils
from utils_LATTE import get_loaders
from modeling_LATTE import LATTE

# Model configurations
from config_LATTE import Config
config = Config()

from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataloader, model, utils):
    """ Train the model """
    tb_writer = SummaryWriter()

    print(args.num_train_epochs)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    print(t_total)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Check if saved optimizer states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")):
        # Load in optimizer states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_batch_size * args.n_gpu
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and args.resume:
        # set global_step to gobal_step of last saved checkpoint from model path
        # global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        global_step = int(args.model_name_or_path.split("/")[-2].split("-")[-1])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(train_dataloader), disable=False)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # print(batch[0].size())
            # print(batch[1].size())
            # print(batch[2].size())
            # print(batch[3].size())
            # print(batch[4].size())
            inputs = {"mention": batch[0],
                      "candidates": batch[1],
                      "target_candidate": batch[2],
                      "mention_type": batch[3],
                      "candidate_types": batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            print(loss)
            loss.backward()
            print("After backward")

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                tb_writer.add_scalar("loss", (tr_loss - logging_loss), global_step)
                logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print("Before saving model ...")
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    #model_to_save.save_pretrained(output_dir)
                    output_file = os.path.join(output_dir, 'model.ckpt')
                    torch.save(model_to_save.state_dict(), output_file)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataloader, model, utils, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
         os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    results = {}

    p_1 = 0
    map = 0
    nb_samples = 0
    nb_normalized = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"mention": batch[0],
                      "candidates": batch[1],
                      "target_candidate": batch[2],
                      "mention_type": batch[3],
                      "candidate_types": batch[4]}
            outputs = model(**inputs)
            tmp_eval_loss, ranks = outputs
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        scores = ranks.detach().cpu().numpy()
        sorted_scores = np.flip(np.argsort(scores), axis=1)
        target_candidate_ids = inputs["target_candidate"].reshape(-1).detach().cpu().numpy()

        for i, score in enumerate(sorted_scores):
            if target_candidate_ids[i] != -100:
                rank = np.where(score == target_candidate_ids[i])[0][0] + 1
                map += 1 / rank
                if rank == 1:
                    p_1 += 1
                    nb_normalized += 1
        nb_samples += scores.shape[0]

    eval_loss = eval_loss / nb_eval_steps

    # Unnormalized precision
    p_1_unnormalized = p_1 / nb_samples
    map_unnormalized = map / nb_samples

    # Normalized precision
    p_1_normalized = p_1 / nb_normalized
    map_normalized = map / nb_normalized

    print("P@1 Unnormalized = ", p_1_unnormalized)
    print("MAP Unnormalized = ", map_unnormalized)
    print("P@1 Normaliized = ", p_1_normalized)
    print("MAP Normalized = ", map_normalized)

    results["P@1"] = p_1_unnormalized
    results["MAP"] = map_unnormalized

    return results


def main():
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--do_train', action='store_true', default=False, help='Switch on for training')
    mode_group.add_argument('--do_eval', action='store_true', default=False, help='Switch on for inference')
    parser.add_argument("--resume", type=bool, default=False, help="On this to resume training")
    parser.add_argument("--model_name_or_path",
                        default=None, type=str, required=True, help="Path to pre-trained model")
    parser.add_argument('--data_dir',
                        type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir",
                        default=None, type=str, required=True, help="Path to pre-trained model")
    parser.add_argument('--max_len', type=int, default=32, help="Max sequence length")
    parser.add_argument('--max_num_candidates', type=int, default=10, help="Number of candidates to consider")
    parser.add_argument('--per_gpu_batch_size', type=int, default=8, help="Training and eval batch size ")
    parser.add_argument('--num_workers', type=int, default=4, help="NUmber of workers for dataloader")
    parser.add_argument('--num_train_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-04, help="Optimizer learning rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Accumulate gradient for steps")
    parser.add_argument('--save_steps', type=int, default=500, help='Number of epochs to wait before early stopping')
    parser.add_argument('--eval_interval', type=int, default=2, help='Evaluation after interval of epochs')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max grad norm")

    args = parser.parse_args()

    config.mention_len = args.max_len
    config.candidate_len = args.max_len // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # DataUtils
    utils = DataUtils()
    utils.build_vocabulary(args.data_dir)
    utils.build_char_vocabulary(args.data_dir)
    glove_embedding_path = './glove_embeddings.json'
    utils.set_glove_embeddings(glove_embedding_path)
    print("#### Data Utils Summary ####")
    print("Total number of words = {}".format(len(utils.vocab)))
    print("Total number of types = {}".format(len(utils.types)))
    print("Total number of characters = {}".format(utils.num_char))
    print("Number of words with GloVe embeddings = {}".format(len(utils.glove_embeddings)))

    # Model
    model = LATTE(config, utils, args.device)
    model.to(device)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Data
    print(" **** Loading datasets ...")
    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_dataloader, dev_dataloader, test_dataloader = get_loaders(args.data_dir,
                                                           utils,
                                                           args.max_len,
                                                           args.max_num_candidates,
                                                           batch_size,
                                                           args.num_workers)
    print("#### Dataset Summary ####")
    print("Num. train batches = {}".format(len(train_dataloader)))
    print("Num. dev batches = {}".format(len(dev_dataloader)))
    print("Num. test batches = {}".format(len(test_dataloader)))

    if args.do_train:
        train(args, train_dataloader, model, utils)
    elif args.do_eval:
        evaluate(args, test_dataloader, model, utils)


if __name__ == '__main__':
    main()
