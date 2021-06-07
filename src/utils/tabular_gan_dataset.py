
from torch.utils.data import Dataset
import torch 

__all__ = ["TabularGANDataSet"]

class TabularGANDataSet(Dataset):
    """Students Performance dataset."""

    def __init__(self, state,action,reward):
        """Initializes instance of class StudentsPerformanceDataset.

        Args:
            state  : pd.DataFrame
            action : pd.DataFrame 
            reward : pd.DataFrame
        
        next_state is not defined... 
        """

        # Save target and predictors
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = state # .sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.state)
    
    def get_label(self,idx) :
        ## supported by label only integer
        return int(self.action.iloc[idx].values)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.state.iloc[idx].values, 
        self.action.iloc[idx].values, 
        self.reward.iloc[idx].values,
        self.next_state.iloc[idx].values]