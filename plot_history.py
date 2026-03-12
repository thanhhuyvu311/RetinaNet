import pandas as pd
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
    base_dir = '/home/huy/Documents/de_tai_tot_nghiep/object_detect'
    hist_store_dir = os.path.join(base_dir,'hist_store')

    hist_df = pd.read_csv(hist_store_dir+'/100epochs_sgd_csv.csv')
    hist_df_adam = pd.read_csv(hist_store_dir+'/100_adam_auto_csv.csv')
    fig,ax = plt.subplots(2,2,figsize = (10,8))

    epochs = range(len(hist_df['loss']))

    #--loss--
    loss = hist_df['loss']
    val_loss = hist_df['val_loss']
    loss_adam = hist_df_adam['loss']
    val_loss_adam = hist_df_adam['val_loss']
    #--plot--

    ax[0,0].plot(epochs,loss,label='train_loss_sgd')
    ax[0,0].plot(epochs,val_loss,label='validation_loss_sgd')
    ax[0,0].set_title('LOSS_SGD')
    ax[0,0].legend()
    ax[0,1].plot(epochs, loss_adam, label='train_loss_adam')
    ax[0,1].plot(epochs, val_loss_adam, label='validation_loss_adam')
    ax[0,1].set_title('LOSS_ADAM')
    ax[0,1].legend()

    # --- Thêm giá trị cuối cùng cho SGD ---
    last_loss_sgd = loss.iloc[-1]
    last_val_loss_sgd = val_loss.iloc[-1]

    ax[0, 0].annotate(f'{last_loss_sgd:.4f}', xy=(epochs[-1], last_loss_sgd),
                      textcoords="offset points", xytext=(5, 0), va='top', color='blue', fontsize=9)
    ax[0, 0].annotate(f'{last_val_loss_sgd:.4f}', xy=(epochs[-1], last_val_loss_sgd),
                      textcoords="offset points", xytext=(5, 0), va='center', color='orange', fontsize=9)

    # --- Thêm giá trị cuối cùng cho ADAM ---
    last_loss_adam = loss_adam.iloc[-1]
    last_val_loss_adam = val_loss_adam.iloc[-1]

    ax[0, 1].annotate(f'{last_loss_adam:.4f}', xy=(epochs[-1], last_loss_adam),
                      textcoords="offset points", xytext=(5, 0), va='top', color='blue', fontsize=9)
    ax[0, 1].annotate(f'{last_val_loss_adam:.4f}', xy=(epochs[-1], last_val_loss_adam),
                      textcoords="offset points", xytext=(5, 0), va='center', color='orange', fontsize=9)
    plt.tight_layout()
    plt.show()
