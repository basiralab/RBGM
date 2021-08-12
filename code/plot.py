import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.manifold import TSNE
import numpy as np 
d = Path(__file__).resolve().parents[1]



def plot(loss, title, losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig(str(d) + '/plots/' + title + '.png')
    plt.close()


def visualization(data, generated_data,title):
    print("Start t_SNE")
    anal_sample_no = min(1000,len(data))
    idx = np.random.permutation(len(data))[:anal_sample_no]
    
    data = np.asarray(data)
    generated_data = np.asarray(generated_data)
    
    data = data[idx]
    generated_data = generated_data[idx]
    print(data.shape)
    print(generated_data.shape)
    no,seq_len, dim = data.shape
    
    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(data[0,:,:],1),[1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1),[1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                      np.reshape(np.mean(data[i,:,:],1),[1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
            
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    print(np.shape(prep_data))
    print(np.shape(prep_data_hat))
    print(np.shape(prep_data_final))
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
    print(np.shape(tsne_results))
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Ground truth")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Predicted")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(str(d) + '/tsne/' + title + '.png')
    plt.close()
    print("End t-SNE")