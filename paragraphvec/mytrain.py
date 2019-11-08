#%%
import time
from sys import float_info, stdout
import torch
from torch.optim import Adam
#%%
from paragraphvec.data import load_dataset, NCEData
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DM, DBOW
from paragraphvec.utils import save_training_state
#%%
cur = None 
# %%
def start(data_file_name,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          model_ver='dbow',
          context_size=0,
          vec_combine_method='sum',
          save_all=False,
          generate_plot=True,
          max_generated_batches=5,
          num_workers=1):
    
    if model_ver not in ('dm', 'dbow'):
        raise ValueError("Invalid version of the model")

    model_ver_is_dbow = model_ver == 'dbow'

    if model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    if not model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word "
                             "vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")
    losses = [] 
    dataset = load_dataset(data_file_name)
    nce_data = NCEData(
        dataset,
        batch_size,
        context_size,
        num_noise_words,
        max_generated_batches,
        num_workers)
    nce_data.start()

    try:
        print("num_batches",len(nce_data))
        print(type(nce_data))
        losses =_run(data_file_name, dataset, nce_data.get_generator(), len(nce_data),
             nce_data.vocabulary_size(), context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all, generate_plot, model_ver_is_dbow)
    except KeyboardInterrupt:
        nce_data.stop()
    return losses 
#%%
def _run(data_file_name,
         dataset,
         data_generator,
         num_batches,
         vocabulary_size,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver,
         vec_combine_method,
         save_all,
         generate_plot,
         model_ver_is_dbow):

    if model_ver_is_dbow:
        model = DBOW(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)
    else:
        model = DM(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents.".format(len(dataset)))
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")
    losses =[]
    best_loss = float("inf")
    prev_model_file_path = None

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []
        for ind_i,batch_i in enumerate(range(num_batches)):
            batch = next(data_generator)
            if torch.cuda.is_available():
                batch.cuda_()

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)
            else:
                x = model.forward(
                    batch.context_ids,
                    batch.doc_ids,
                    batch.target_noise_ids)

            x = cost_func.forward(x)

            loss.append(x.item())
            model.zero_grad()
            x.backward()
            optimizer.step()
            stdout.write(str(ind_i)+" ")
            stdout.flush()
        _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:f}s) - loss: {:.4f}".format(epoch_total_time, loss))
        losses.append(loss.data.tolist())
    return losses 

def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1))
    stdout.write(" - {:f}%".format(progress))
    stdout.flush()

#%%
data_file_name = 'filtered_text_as_seen.csv'
num_noise_words = 2
vec_dim = 100
num_epochs = 100
batch_size = 10000
lr = 0.001
losses = start(data_file_name, num_noise_words, vec_dim, num_epochs, batch_size,lr)

# %%
losses


# %%
import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))),losses)


# %%
