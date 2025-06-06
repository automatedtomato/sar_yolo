import matplotlib.pyplot as plt
import seaborn as sns

def learning_curve(train_history: dict[str: list], val_history: dict[str: list], save_fig: bool = False, save_path: str = None):
    """
    Plot learning curves
    
    Args:
        train_history (dict[str: list]): Training history
        val_history (dict[str: list]): Validation history
        save_fig (bool, optional): Whether to save the figure. Defaults to False.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    
    plt.figure(figsize=(15, 8))
    
    for i, (k, _) in enumerate(train_history.items()):
        plt.subplot(2, 3, i+1)
        sns.lineplot(train_history[k], label=f"train_{k}")
        sns.lineplot(val_history[k], label=f"val_{k}")
        plt.tight_layout()
    
    if save_fig:
        if save_path is None:
            raise ValueError('save_path must be specified if save_fig is true')
        if save_path.split('.')[-1] != 'png':
            raise ValueError('save_path must be a png file')
        
        plt.savefig(save_path)
            
    plt.legend()
    plt.show()
    