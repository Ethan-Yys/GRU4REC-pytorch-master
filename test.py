import pandas as pd
import numpy as np
item_ids = [6,4,8,2,8]
item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids)
print(item2idx)