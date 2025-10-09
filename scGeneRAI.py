import os
import re
import copy
import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import permutations
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR


class Dataset_train(Dataset):
    '''
    Random-masking training dataset for denoising-style self-supervision.

    Args:
        data (torch.Tensor): Tensor of shape (n_samples, n_features) with input values in [0, 1].
        interval (tuple[float, float]): Range to sample masking probability p ~ Uniform(interval).

    Returns (per item):
        masked_concat (torch.Tensor): Concatenation of masked x and masked (1 - x); shape (2*n_features,).
        mask (torch.Tensor): Binary mask (1 = kept, 0 = masked); shape (n_features,).
        full_data (torch.Tensor): Original unmasked vector; shape (n_features,).
    '''
    def __init__(self, data, interval=(0.01, 0.99)):
        self.nsamples, self.nfeatures = data.shape
        self.data = data
        self.l = data.shape[0]
        self.interval = interval

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        p = np.random.uniform(self.interval[0], self.interval[1])
        full_data = self.data[idx, :]
        mask = (tc.rand_like(full_data) < p) * 1.0
        x_1, x_2 = full_data.clone(), 1 - full_data.clone()
        x_1[mask == 0], x_2[mask == 0] = 0, 0
        return tc.cat((x_1, x_2), axis=0), mask, full_data


class Dataset_LRP(Dataset):
    '''
    Dataset for Layer-wise Relevance Propagation (LRP) sampling.

    Args:
        data (torch.Tensor): Tensor of shape (n_samples, n_features).
        target_id (int): Index of the target feature to explain.
        sample_id (int): Index of the specific sample to probe.
        maskspersample (int): Number of random masks to generate.

    Returns (per item):
        masked_concat (torch.Tensor): Concatenated masked inputs; shape (2*n_features,).
        mask (torch.Tensor): Binary mask with target forced to 0; shape (n_features,).
        full_data (torch.Tensor): Original unmasked vector of the selected sample; shape (n_features,).
    '''
    def __init__(self, data, target_id, sample_id, maskspersample=10000):
        self.nsamples, self.nfeatures = data.shape
        self.data = data
        self.l = data.shape[0]
        self.target_id = target_id
        self.sample_id = sample_id
        self.maskspersample = maskspersample

    def __len__(self):
        return self.maskspersample

    def __getitem__(self, idx):
        full_data = self.data[self.sample_id, :]
        mask = (tc.rand_like(full_data) < 0.5) * 1.0
        mask[self.target_id] = 0
        x_1, x_2 = full_data.clone(), 1 - full_data.clone()
        x_1[mask == 0], x_2[mask == 0] = 0, 0
        return tc.cat((x_1, x_2), axis=0), mask, full_data


def compute_LRP(neuralnet, test_set, target_id, sample_id, batch_size, device):
    '''
    Compute Layer-wise Relevance Propagation (LRP) scores for a given target and sample.

    Args:
        neuralnet (nn.Module): Trained network supporting .relprop(R).
        test_set (torch.Tensor): All samples tensor (n_samples, n_features).
        target_id (int): Target feature index to explain.
        sample_id (int): Sample index to probe within test_set.
        batch_size (int): DataLoader batch size for LRP masking.
        device (torch.device): Device to run on.

    Returns:
        LRP_scaled (np.ndarray): Scaled relevance per input feature; shape (n_features,).
        error (np.ndarray): MSE per-batch element vs. ground truth target.
        y (float): Mean ground-truth value of target over the sampled batch.
        y_pred (float): Mean predicted value of target over the sampled batch.
        full_data_sample (np.ndarray): Original feature values for the chosen sample.
    '''
    criterion = nn.MSELoss()
    testset = Dataset_LRP(test_set, target_id, sample_id)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    neuralnet.to(device).eval()

    masked_data, mask, full_data = next(iter(testloader))
    masked_data, mask, full_data = masked_data.to(device), mask.to(device), full_data.to(device)
    pred = neuralnet(masked_data)

    error = criterion(pred.detach()[:, target_id], full_data.detach()[:, target_id]).cpu().numpy()
    y = full_data.detach()[:, target_id].cpu().mean().numpy()
    y_pred = pred.detach()[:, target_id].cpu().mean().numpy()

    R = tc.zeros_like(pred)
    R[:, target_id] = pred[:, target_id].clone()
    a = neuralnet.relprop(R)
    LRP_sum = a.sum(dim=0)
    LRP_unexpanded = 0.5 * (LRP_sum[: LRP_sum.shape[0] // 2] + LRP_sum[LRP_sum.shape[0] // 2 :])

    mask_sum = mask.sum(dim=0).float()
    LRP_scaled = LRP_unexpanded / mask_sum
    LRP_scaled = tc.where(tc.isnan(LRP_scaled), tc.tensor(0.0).to(device), LRP_scaled)

    full_data_sample = full_data[0, :].cpu().detach().numpy().squeeze()
    return LRP_scaled.cpu().numpy(), error, y, y_pred, full_data_sample


def calc_all_paths(
    neuralnet,
    test_data,
    sample_id,
    sample_name,
    featurenames,
    target_gene_range,
    PATH,
    batch_size=100,
    LRPau=True,
    device=tc.device('cpu'),
    source_gene_indices=None
):
    '''
    Run LRP for a set of target indices on a single sample and save per-source attributions.

    Args:
        neuralnet (nn.Module): Trained network supporting .relprop(R).
        test_data (torch.Tensor): Tensor (n_samples, n_features).
        sample_id (int): Index of the sample to evaluate.
        sample_name (str): Human-readable sample identifier; used in output file naming.
        featurenames (Iterable[str]): Names for all features/inputs.
        target_gene_range (Iterable[int] | int): Target indices (or int N => range(N)).
        PATH (str): Output directory; results saved under PATH/results/.
        batch_size (int): Batch size for LRP masking loader.
        LRPau (bool): Unused flag preserved for compatibility.
        device (torch.device): Device to run on.
        source_gene_indices (Iterable[int] | int | None): Indices to report as sources.

    Side effects:
        Writes CSV: PATH/results/LRP_{sample_id}_{sample_name}.csv
    '''
    frames = []

    if isinstance(target_gene_range, (int, np.integer)):
        target_iter = range(target_gene_range)
    else:
        target_iter = list(target_gene_range)

    if source_gene_indices is None:
        if isinstance(target_gene_range, (int, np.integer)):
            source_idx = list(range(target_gene_range))
        else:
            source_idx = list(range(test_data.shape[1]))
    else:
        if isinstance(source_gene_indices, (int, np.integer)):
            source_idx = list(range(source_gene_indices))
        else:
            source_idx = list(source_gene_indices)

    feat_names_array = np.array(featurenames)

    for target in target_iter:
        LRP_value, error, y, y_pred, full_data_sample = compute_LRP(
            neuralnet, test_data, target, sample_id, batch_size=batch_size, device=device
        )
        frame = pd.DataFrame({
            'LRP': np.asarray(LRP_value)[source_idx],
            'source_gene': feat_names_array[source_idx],
            'target_gene': feat_names_array[target],
            'sample_name': sample_name,
            'error': error,
            'y': y,
            'y_pred': y_pred,
            'inpv': np.asarray(full_data_sample)[source_idx],
        })
        frames.append(frame)

    end_frame = pd.concat(frames, ignore_index=True)
    result_dir = os.path.join(PATH, "results")
    os.makedirs(result_dir, exist_ok=True)
    safe_sample = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_name))
    out_path = os.path.join(result_dir, f"LRP_{sample_id}_{safe_sample}.csv")
    end_frame.to_csv(out_path, index=False)


class LogCoshLoss(nn.Module):
    '''
    Smooth robust loss: mean(log(cosh(y_true - y_pred))).

    Numerically stable variant with a small constant for cosh argument.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return tc.mean(tc.log(tc.cosh(ey_t + 1e-12)))


class LRP_Linear(nn.Module):
    '''
    Linear layer with LRP gamma-rule relevance backpropagation.

    Args:
        inp (int): Input dimension.
        outp (int): Output dimension.
        gamma (float): Gamma factor for positive/negative weight stabilization.
        eps (float): Numerical stabilizer for division.

    Notes:
        - During eval mode, forward() stores activations per self.iteration key in A_dict.
        - relprop(R) performs relevance redistribution using cloned and reparameterized layers.
    '''
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):
        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device
        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)

        zpp = self.newlayer(1).forward(Ap)
        zmm = self.newlayer(-1, no_bias=True).forward(Am)
        zmp = self.newlayer(1, no_bias=True).forward(Am)
        zpm = self.newlayer(-1).forward(Ap)

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sp).sum().backward()
        cmp = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sm).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data
        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):
        '''
        Create a cloned Linear layer with rho-transformed parameters.

        Args:
            sign (int): +1 or -1; selects positive or negative projection rule.
            no_bias (bool): If True, sets bias to zero.

        Returns:
            nn.Linear: Reparameterized linear layer copy.
        '''
        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0)
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0)

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new


class LRP_ReLU(nn.Module):
    '''
    ReLU layer that passes relevance unchanged in relprop.
    '''
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class NN(nn.Module):
    '''
    Feed-forward network with LRP-aware Linear+ReLU blocks.

    Args:
        inp (int): Input dimension.
        outp (int): Output dimension.
        hidden (int): Hidden layer width.
        hidden_depth (int): Number of hidden blocks to append (each: Linear -> ReLU).

    Methods:
        relprop(R): Propagate relevance R backward through the network (eval mode only).
    '''
    def __init__(self, inp, outp, hidden, hidden_depth):
        super(NN, self).__init__()
        self.layers = nn.Sequential(LRP_Linear(inp, hidden), LRP_ReLU())
        for i in range(hidden_depth):
            self.layers.add_module('LRP_Linear' + str(i + 1), LRP_Linear(hidden, hidden))
            self.layers.add_module('LRP_ReLU' + str(i + 1), LRP_ReLU())
        self.layers.add_module('LRP_Linear_last', LRP_Linear(hidden, outp))

    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        '''
        Relevance backpropagation wrapper that iterates modules in reverse.
        '''
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R


class OneHotter:
    '''
    Lightweight one-hot encoder that preserves train-time categorical levels.

    Methods:
        make_one_hot_new(df): Fit+transform; stores per-column levels.
        make_one_hot(df): Transform using stored levels.
    '''
    def __init__(self):
        pass

    def make_one_hot_new(self, descriptors):
        '''
        Fit and transform categorical descriptors into one-hot columns.

        Args:
            descriptors (pd.DataFrame): Categorical columns.

        Returns:
            pd.DataFrame: One-hot encoded frame with "col=value" column names.
        '''
        columns = []
        self.level_dict = {}
        for col in descriptors.columns:
            sel_col = descriptors[col]
            levels = sel_col.unique()
            self.level_dict[col] = levels
            one_hot = (np.array(sel_col)[:, None] == levels[None, :]) * 1.0
            colnames = [col + '=' + level for level in levels]
            one_hot_frame = pd.DataFrame(one_hot, columns=colnames)
            columns.append(one_hot_frame)
        return pd.concat(columns, axis=1)

    def make_one_hot(self, descriptors):
        '''
        Transform descriptors into one-hot using fitted levels.

        Args:
            descriptors (pd.DataFrame): Categorical columns.

        Returns:
            pd.DataFrame: One-hot encoded frame aligned to stored levels.
        '''
        columns = []
        for col in descriptors.columns:
            sel_col = descriptors[col]
            levels = self.level_dict[col]
            one_hot = (np.array(sel_col)[:, None] == levels[None, :]) * 1.0
            colnames = [col + '=' + level for level in levels]
            one_hot_frame = pd.DataFrame(one_hot, columns=colnames)
            columns.append(one_hot_frame)
        return pd.concat(columns, axis=1)


def train(neuralnet, train_data, test_data, epochs, lr, batch_size, lr_decay, device_name):
    '''
    Train loop with SGD + momentum, exponential LR decay, and log-cosh loss.
    Periodically evaluates on a masked version of the test set and returns the
    history needed for early stopping.

    Args:
        neuralnet (nn.Module): Model to train.
        train_data (torch.Tensor): Training tensor (n_train, n_features).
        test_data (torch.Tensor): Held-out tensor (n_test, n_features).
        epochs (int): Number of epochs.
        lr (float): Initial learning rate.
        batch_size (int): Batch size.
        lr_decay (float): Exponential decay factor per epoch.
        device_name (str): Torch device name (e.g., 'cpu', 'cuda:0').

    Returns:
        testlosses (torch.Tensor): Collected test losses at checkpoints.
        epoch_list (list[int]): Epoch numbers corresponding to testlosses.
        network_list (list[dict]): State dict snapshots for early stopping.
    '''
    device = tc.device(device_name)
    nsamples, nfeatures = train_data.shape
    optimizer = tc.optim.SGD(neuralnet.parameters(), lr=lr, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    criterion = LogCoshLoss()
    testlosses, epoch_list, network_list = [], [], []
    neuralnet.train().to(device)

    for epoch in tqdm(range(epochs)):
        if epoch < 5:
            optimizer.param_groups[0]['lr'] = lr / 5 * (epoch + 1)

        trainset = Dataset_train(train_data)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        for masked_data, mask, full_data in trainloader:
            masked_data = masked_data.to(device)
            mask = mask.to(device)
            full_data = full_data.to(device)
            optimizer.zero_grad()
            pred = neuralnet(masked_data)
            loss = criterion(pred[mask == 0], full_data[mask == 0])
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            neuralnet.eval()
            testset = Dataset_train(test_data)
            traintestset = Dataset_train(train_data)
            testloader = DataLoader(testset, batch_size=test_data.shape[0], shuffle=False)
            traintestloader = DataLoader(traintestset, batch_size=test_data.shape[0], shuffle=False)

            for masked_data, mask, full_data in testloader:
                masked_data = masked_data.to(device)
                mask = mask.to(device)
                full_data = full_data.to(device)
                with tc.no_grad():
                    pred = neuralnet(masked_data)
                testloss = criterion(pred[mask == 0], full_data[mask == 0])
                testlosses.append(testloss)
                epoch_list.append(epoch)
                network_list.append(neuralnet.state_dict())
                break

            for masked_data, mask, full_data in traintestloader:
                masked_data = masked_data.to(device)
                mask = mask.to(device)
                full_data = full_data.to(device)
                with tc.no_grad():
                    pred = neuralnet(masked_data)
                traintestloss = criterion(pred[mask == 0], full_data[mask == 0])
                break
            neuralnet.train()

    return tc.tensor(testlosses), epoch_list, network_list


class scGeneRAI:
    '''
    High-level wrapper for training the LRP-capable network on (optionally extended)
    single-cell matrices and exporting attribution networks per sample via LRP.

    Workflow:
        - fit(): train model with optional categorical descriptors (one-hot).
        - predict_networks(): run LRP for selected targets/samples and save CSVs.
    '''
    def __init__(self):
        pass

    def fit(
        self,
        data,
        nepochs,
        model_depth,
        lr=2e-2,
        batch_size=5,
        lr_decay=0.995,
        descriptors=None,
        early_stopping=True,
        device_name='cpu'
    ):
        '''
        Fit the model on data with optional categorical descriptors.

        Args:
            data (pd.DataFrame): Continuous features per sample (index = samples).
            nepochs (int): Max epochs to train.
            model_depth (int): Number of hidden Linear+ReLU blocks.
            lr (float): Learning rate.
            batch_size (int): Mini-batch size.
            lr_decay (float): Exponential LR decay factor per epoch.
            descriptors (pd.DataFrame | None): Categorical covariates to one-hot and append.
            early_stopping (bool): If True, restore best snapshot by test loss.
            device_name (str): Device string for torch.

        Side effects:
            Prints the selected epoch and test loss after training.
        '''
        self.simple_features = data.shape[1]
        if descriptors is not None:
            self.onehotter = OneHotter()
            one_hot_descriptors = self.onehotter.make_one_hot_new(descriptors)
            self.data = pd.concat([data, one_hot_descriptors], axis=1)
        else:
            self.data = data

        self.nsamples, self.nfeatures = self.data.shape
        self.hidden = 2 * self.nfeatures
        self.depth = model_depth

        self.sample_names = self.data.index
        self.feature_names = self.data.columns
        self.data_tensor = tc.tensor(np.array(self.data)).float()

        self.nn = NN(2 * (self.nfeatures), self.nfeatures, self.hidden, self.depth)

        tc.manual_seed(0)
        all_ids = tc.randperm(self.nsamples)
        self.train_ids, self.test_ids = all_ids[: self.nsamples // 10 * 9], all_ids[self.nsamples // 10 * 9 :]
        testlosses, epoch_list, network_list = train(
            self.nn,
            self.data_tensor[self.train_ids],
            self.data_tensor[self.test_ids],
            nepochs,
            lr=lr,
            batch_size=batch_size,
            lr_decay=lr_decay,
            device_name=device_name
        )

        if early_stopping:
            mindex = tc.argmin(testlosses)
            self.actual_testloss = testlosses[mindex]
            min_network = network_list[mindex]
            self.epochs_trained = epoch_list[mindex]
            self.nn = NN(2 * (self.nfeatures), self.nfeatures, self.hidden, self.depth)
            self.nn.load_state_dict(min_network)
        else:
            self.epochs_trained = nepochs
            self.actual_testloss = testlosses[-1]

        print('the network trained for {} epochs (testloss: {})'.format(self.epochs_trained, self.actual_testloss))

    def predict_networks(
        self,
        data,
        descriptors=None,
        LRPau=True,
        remove_descriptors=True,
        device_name='cpu',
        PATH='.',
        targets=None,
        fail_on_missing=False
    ):
        '''
        Run LRP per sample and target(s) to export source->target attributions.

        Args:
            data (pd.DataFrame): Continuous features per sample.
            descriptors (pd.DataFrame | None): Categorical covariates for one-hot transform.
            LRPau (bool): Unused flag preserved for compatibility.
            remove_descriptors (bool): If True, restrict sources/targets to base features only.
            device_name (str): Torch device name.
            PATH (str): Output directory; CSVs written to PATH/results/.
            targets (list[str] | None): Feature names to treat as targets; defaults to base sources.
            fail_on_missing (bool): Raise if requested targets are missing/illegal.

        Side effects:
            Writes one CSV per sample with columns:
            ['LRP','source_gene','target_gene','sample_name','error','y','y_pred','inpv'].
        '''
        if not os.path.exists(PATH + '/results/'):
            os.makedirs(PATH + '/results/')

        if descriptors is not None:
            one_hot_descriptors = self.onehotter.make_one_hot(descriptors)
            assert one_hot_descriptors.shape[0] == data.shape[0], \
                f"descriptors ({one_hot_descriptors.shape[0]}) need to have same sample size as data ({data.shape[0]})"
            data_extended = pd.concat([data, one_hot_descriptors], axis=1)
        else:
            data_extended = data

        nsamples_LRP, nfeatures_LRP = data_extended.shape
        assert nfeatures_LRP == self.nfeatures, \
            f"neural network has been trained on {self.nfeatures} input features, now there are {nfeatures_LRP}"

        sample_names_LRP = data_extended.index
        feature_names_LRP = data_extended.columns
        data_tensor_LRP = tc.tensor(np.array(data_extended)).float()

        if remove_descriptors:
            sf = self.simple_features
            if isinstance(sf, (int, np.integer)):
                source_indices = list(range(sf))
            else:
                source_indices = list(sf)
        else:
            source_indices = list(range(data_tensor_LRP.shape[1]))

        if targets is None:
            target_indices = source_indices if remove_descriptors else list(range(data_tensor_LRP.shape[1]))
        else:
            name_to_idx = {name: i for i, name in enumerate(feature_names_LRP)}
            missing, illegal, target_indices = [], [], []
            for tname in targets:
                if tname not in name_to_idx:
                    missing.append(tname)
                    continue
                idx = name_to_idx[tname]
                if idx in (source_indices if remove_descriptors else range(data_tensor_LRP.shape[1])):
                    target_indices.append(idx)
                else:
                    illegal.append(tname)
            if missing:
                msg = "[predict_networks] Targets not found in columns: " + ", ".join(missing)
                if fail_on_missing:
                    raise ValueError(msg)
                else:
                    print("WARN:", msg)
            if illegal and remove_descriptors:
                msg = "[predict_networks] Not allowed with remove_descriptors=True: " + ", ".join(illegal)
                if fail_on_missing:
                    raise ValueError(msg)
                else:
                    print("WARN:", msg)
            seen = set()
            target_indices = [x for x in target_indices if not (x in seen or seen.add(x))]
            if len(target_indices) == 0:
                raise ValueError("No valid targets resolved. Check names and remove_descriptors.")

        device = tc.device(device_name)
        for sample_id, sample_name in enumerate(sample_names_LRP):
            calc_all_paths(
                self.nn,
                data_tensor_LRP,
                sample_id,
                sample_name,
                feature_names_LRP,
                target_gene_range=target_indices,
                PATH=PATH,
                batch_size=100,
                LRPau=LRPau,
                device=device,
                source_gene_indices=source_indices
            )


