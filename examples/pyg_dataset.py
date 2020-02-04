import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.item_num = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data.csv']

    @property
    def processed_file_names(self):
        return ['yoochoose_click_binary_10k_sess.dataset']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        item_num = df.item_id.max() + 1  # 数据集中item的数量

        data_list = []

        # process by session_id，以item为节点（重新编号），以原item_id为特征
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)  # 删除索引列
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']]\
                                 .sort_values('sess_item_id')\
                                 .item_id.drop_duplicates()\
                                 .values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes,
                                       target_nodes], dtype=torch.long)
            x = node_features

            y = torch.LongTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices, item_num), self.processed_paths[0])
