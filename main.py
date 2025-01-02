from sklearn.metrics import f1_score
from models import Encoder, Classifier
from utils import *
import params

def main(parameter):
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')

        path, uda_method, src, tgt, batch_size = (parameter[k] for k in
                                                  ['path', 'uda_method', 'source', 'target', 'batch_size'])

        print(f'{"PE":13s} {"UDA":13s} {"f1-score":10s}')

        for strategy in ['Deterministic', 'Gaussian']:
            for uda in ['source_only', uda_method]:
                path_e = os.path.join(path, f"{src}_to_{tgt}_{strategy}_{uda}_encoder.pt")
                path_c = os.path.join(path,
                                      f"{src}_to_{tgt}_{strategy}_{'source_only' if uda == 'adda' else uda}_classifier.pt")
                path_c2 = os.path.join(path, f"{src}_to_{tgt}_{strategy}_{uda}_classifier2.pt") if uda == 'mcd' else None

                net_e = Encoder(**parameter['param_e'], strategy=strategy).to(DEVICE)
                net_e.load_state_dict(torch.load(path_e))
                net_e.eval()

                net_c = Classifier(**parameter['param_c']).to(DEVICE)
                net_c.load_state_dict(torch.load(path_c))
                net_c.eval()

                net_c2 = None
                if path_c2:
                    net_c2 = Classifier(**parameter['param_c']).to(DEVICE)
                    net_c2.load_state_dict(torch.load(path_c2))
                    net_c2.eval()

                src_dataloader = load_dataloader(os.path.join(path, f"{src}_test.pt"), batch_size)
                tgt_dataloader = load_dataloader(os.path.join(path, f"{tgt}_test.pt"), batch_size)

                src_features, src_class_labels = get_feature(net_e, src_dataloader, 0)
                tgt_features, tgt_class_labels = get_feature(net_e, tgt_dataloader, 0)

                src_features, tgt_features = (torch.tensor(f, dtype=torch.float32).to(DEVICE) for f in
                                              [src_features, tgt_features])
                binary = net_c.out_node == 1

                with torch.no_grad():

                    src_output = compute_output(net_c, src_features, net_c2)
                    tgt_output = compute_output(net_c, tgt_features, net_c2)

                    src_pred, tgt_pred = (predict(output, binary).cpu().numpy() for output in [src_output, tgt_output])

                scoring_args = {'average': 'weighted'} if not binary else {}
                src_f1 = f1_score(src_class_labels, src_pred, **scoring_args)
                tgt_f1 = f1_score(tgt_class_labels, tgt_pred, **scoring_args)

                if strategy == 'Deterministic':
                    first = '\u2717'  # cross mark
                else:
                    first = '\u2713' + ' (' + strategy + ')'  # check mark
                if uda == 'source_only':
                    second = '\u2717'
                else:
                    second = '\u2713' + ' (' + uda + ')'
                print(f'{first:13s} {second:13s} src: {src_f1 * 100:.2f} tgt: {tgt_f1 * 100:.2f}')
        print('\n')
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    print('Task1: T1 -> T2')
    main(params.task1)
    print('Task2: T2b -> F')
    main(params.task2)