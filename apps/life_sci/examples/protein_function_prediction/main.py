import argparse
import dgl.function as fn
import torch
import torch.nn as nn

from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch.optim import Adam

from logger import Logger

class GINConv(nn.Module):
    def __init__(self, in_dim):
        super(GINConv, self).__init__()

        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU()
        )

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.edata['he'] = edge_feats
        g.update_all(fn.u_add_e('hv', 'he', 'm'), fn.mean('m', 'hv_new'))
        node_feats = (1 + self.eps) * g.ndata['hv'] + g.ndata['hv_new']

        return self.mlp(node_feats)

class GIN(nn.Module):
    def __init__(self, in_dim=8, num_layer=2, emb_dim=50, num_task=112):
        super(GIN, self).__init__()

        self.node_encoder = nn.Embedding(1, emb_dim)
        self.edge_encoder = nn.Linear(in_dim, emb_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layer):
            self.gnn_layers.append(GINConv(emb_dim))

        self.pred_out = nn.Linear(emb_dim, num_task)

    @property
    def device(self):
        return self.pred_out.weight.device

    def forward(self, g, edge_feats):
        node_types = torch.zeros(g.number_of_nodes()).long().to(self.device)
        node_feats = self.node_encoder(node_types)
        edge_feats = self.edge_encoder(edge_feats)

        # Message passing
        for gnn_layer in self.gnn_layers:
            node_feats = gnn_layer(g, node_feats, edge_feats)

        return self.pred_out(node_feats)

def train(model, graph, train_node_idx, criterion, optimizer):
    model.train()
    edge_feats = graph.edata['feat']
    logits = model(graph, edge_feats)[train_node_idx]

    labels = graph.ndata['labels'][train_node_idx]
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().data.item()

def eval(model, graph, splitted_idx, evaluator):
    model.eval()
    edge_feats = graph.edata['feat']
    with torch.no_grad():
        logits = model(graph, edge_feats)

    labels = graph.ndata['labels'].cpu().numpy()
    logits = logits.detach().cpu().numpy()

    train_score = evaluator.eval({
        "y_true": labels[splitted_idx["train"]],
        "y_pred": logits[splitted_idx["train"]]
    })
    val_score = evaluator.eval({
        "y_true": labels[splitted_idx["valid"]],
        "y_pred": logits[splitted_idx["valid"]]
    })
    test_score = evaluator.eval({
        "y_true": labels[splitted_idx["test"]],
        "y_pred": logits[splitted_idx["test"]]
    })

    return train_score['rocauc'], val_score['rocauc'], test_score['rocauc']

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GIN with DGL for ogbn-proteins')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('-nl', '--num-layers', type=int, default=2,
                        help='Number of GIN layers to use (default: 2)')
    parser.add_argument('-ed', '--embed-dim', type=int, default=50,
                        help='Hidden size in GIN (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    args.dataset = 'ogbn-proteins'

    # Only use CPU for now
    device = torch.device("cpu")

    # Data loading and splitting
    dataset = DglNodePropPredDataset(name=args.dataset)
    print(dataset.meta_info[args.dataset])
    splitted_idx = dataset.get_idx_split()

    # Change the dtype and device of tensors
    graph = dataset.graph[0]
    graph.ndata['labels'] = dataset.labels.float().to(device)
    graph.edata['feat'] = graph.edata['feat'].float().to(device)

    criterion = nn.BCEWithLogitsLoss()
    evaluator = Evaluator(args.dataset)
    logger = Logger(args.runs, args)

    for run in range(args.run):
        model = GIN(num_layer=args.num_layers, emb_dim=args.embed_dim).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs+1):
            loss = train(model, graph, splitted_idx['train'], criterion, optimizer)

            if epoch % args.eval_steps == 0:
                train_score, val_score, test_score = eval(model, graph, splitted_idx, evaluator)
                logger.add_result(run, (train_score, val_score, test_score))
                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_score:.2f}%, '
                          f'Valid: {100 * val_score:.2f}% '
                          f'Test: {100 * test_score:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == '__main__':
    main()
