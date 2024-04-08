import os
from time import time
from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend('Agg')


def plot_age_distribution(
        data, path, console, weighted=False, log=True, save=True):
    stext = ('[bold green]Generating plot for age distribution '
             f'with weighted={weighted}...')
    with console.status(stext):
        start = time()
        plt.rcParams['font.family'] = 'Arial'
        id_percent = 'percent'
        if weighted:
            id_percent += '_weighted'
        x = data.loc[id_percent].index
        y = data.loc[id_percent].values
        plt.figure()
        plt.title('Altersverteilung der Studienpopulation')
        plt.xticks(np.arange(12, 52, 2))
        plt.yticks(np.arange(0, 8, 0.5))
        plt.xlabel('Alter bei Geburt [Jahre]')
        plt.ylabel('Anteil Schwangerschaften [%]')
        plt.bar(x, y, color='#9a0941')
        plt.grid(
            axis='y',
            color='#888a88',
            linestyle=':',
            linewidth=0.5)

        if save:
            fname_trunk = 'plot_age_distribution'
            if weighted:
                fname_trunk += '_weighted'
            fpath = os.path.join(path, fname_trunk)
            plt.savefig(fpath + '.png', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.png')
            plt.savefig(fpath + '.pdf')
            if log:
                console.log(f'Saved file {fpath}.pdf')
            plt.savefig(fpath + '.jpg', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.jpg')
        rtime = time() - start
        if log:
            console.log(
                f'Generated plot age distribution with weighted={weighted}'
                f' in {rtime:.1f} seconds')
        plt.close()


def plot_a(data, path, console, weighted=False, log=True, save=True):
    stext = f'[bold green]Generating plot A with weighted={weighted}...'
    with console.status(stext):
        start = time()
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.subplots_adjust(right=0.65)

        x = np.arange(6)

        apx = '_excl_a_percent'
        if weighted:
            apx += '_weighted'
        y1 = np.array(data.iloc[1][
            [
                'T0_A0-100' + apx,
                'T123_A0-100' + apx,
                'T1_A0-100' + apx,
                'T2_A0-100' + apx,
                'T3_A0-100' + apx,
                'T4_A0-100' + apx
            ]])
        y2 = np.array(data.iloc[1][
            [
                'T0_A0-25' + apx,
                'T123_A0-25' + apx,
                'T1_A0-25' + apx,
                'T2_A0-25' + apx,
                'T3_A0-25' + apx,
                'T4_A0-25' + apx
            ]])
        y3 = np.array(data.iloc[1][
            [
                'T0_A26-35' + apx,
                'T123_A26-35' + apx,
                'T1_A26-35' + apx,
                'T2_A26-35' + apx,
                'T3_A26-35' + apx,
                'T4_A26-35' + apx
            ]])
        y4 = np.array(data.iloc[1][
            [
                'T0_A36-100' + apx,
                'T123_A36-100' + apx,
                'T1_A36-100' + apx,
                'T2_A36-100' + apx,
                'T3_A36-100' + apx,
                'T4_A36-100' + apx
            ]])

        width = 0.2
        plt.bar(x-1.5*width, y1, width, color='#d696ae')
        plt.bar(x-0.5*width, y2, width, color='#ced8ed')
        plt.bar(x+0.5*width, y3, width, color='#f7f9c8')
        plt.bar(x+1.5*width, y4, width, color='#ffe6c9')

        apx = '_excl_a+b+d_percent'
        if weighted:
            apx += '_weighted'
        y1e = np.array(data.iloc[1][
            [
                'T0_A0-100' + apx,
                'T123_A0-100' + apx,
                'T1_A0-100' + apx,
                'T2_A0-100' + apx,
                'T3_A0-100' + apx,
                'T4_A0-100' + apx
            ]])
        y2e = np.array(data.iloc[1][
            [
                'T0_A0-25' + apx,
                'T123_A0-25' + apx,
                'T1_A0-25' + apx,
                'T2_A0-25' + apx,
                'T3_A0-25' + apx,
                'T4_A0-25' + apx
            ]])
        y3e = np.array(data.iloc[1][
            [
                'T0_A26-35' + apx,
                'T123_A26-35' + apx,
                'T1_A26-35' + apx,
                'T2_A26-35' + apx,
                'T3_A26-35' + apx,
                'T4_A26-35' + apx
            ]])
        y4e = np.array(data.iloc[1][
            [
                'T0_A36-100' + apx,
                'T123_A36-100' + apx,
                'T1_A36-100' + apx,
                'T2_A36-100' + apx,
                'T3_A36-100' + apx,
                'T4_A36-100' + apx
            ]])
        plt.bar(x-1.5*width, y1e, width, color='#9a0941')
        plt.bar(x-0.5*width, y2e, width, color='#8296c4')
        plt.bar(x+0.5*width, y3e, width, color='#b3a812')
        plt.bar(x+1.5*width, y4e, width, color='#ed7d31')

        plt.title(('Percentage of pregnancies with at least '
                   'one claimed drug prescription'))
        plt.xticks(x, [
            'Pre-pregnancy',
            'Pregnancy',
            'Trimester 1',
            'Trimester 2',
            'Trimester 3',
            'Post-pregnancy'])
        plt.yticks(np.arange(0, 105, 5))
        plt.ylabel("Percentage of pregnancies [%]")
        plt.grid(axis='y', color='#888a88', linestyle=':', linewidth=0.5)
        plt.legend(
            [
                'All age categories (all)',
                '< 26 years (all)',
                '26-35 years (all)',
                '≥ 36 years (all)',
                'All age categories (with exclusions)',
                '< 26 years (with exclusions)',
                '26-35 years (with exclusions)',
                '≥ 36 years (with exclusions)'
            ],
            loc='center left',
            bbox_to_anchor=(1.04, 0.5)
        )
        if save:
            fname_trunk = 'plot_A'
            if weighted:
                fname_trunk += '_weighted'
            fpath = os.path.join(path, fname_trunk)
            plt.savefig(fpath + '.png', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.png')
            plt.savefig(fpath + '.pdf')
            if log:
                console.log(f'Saved file {fpath}.pdf')
            plt.savefig(fpath + '.jpg', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.jpg')
        rtime = time() - start
        if log:
            console.log(
                f'Generated plot A with weighted={weighted}'
                f' in {rtime:.1f} seconds')
        plt.close()


def plot_c(data, path, console, weighted=False, log=True, save=True):
    stext = f'[bold green]Generating plot C with weighted={weighted}...'
    with console.status(stext):
        start = time()
        plt.rcParams['font.family'] = 'Arial'

        apx = '_excl_a+b+d_percent'
        if weighted:
            apx += '_weighted'

        pdata = data[[
            'T123_A0-25' + apx,
            'T123_A26-35' + apx,
            'T123_A36-100' + apx,
            'T123_A0-100' + apx]]

        r = [0, 1, 2, 3]
        raw_data = {
            '0': pdata.iloc[0],
            '1': pdata.iloc[1],
            '2': pdata.iloc[2],
            '3': pdata.iloc[3],
            '4': pdata.iloc[4],
            'ge5': pdata.iloc[5]
        }

        barWidth = 0.70
        names = (
            '< 26 years',
            '26-35 years',
            '≥ 36 years',
            'All age categories')
        plt.rcParams["figure.figsize"] = (10, 5)
        plt.subplots_adjust(right=0.7)
        plt.bar(
            r,
            raw_data['0'],
            color='#8296c4',
            edgecolor='white',
            width=barWidth,
            linewidth=0)
        plt.bar(
            r,
            raw_data['1'],
            bottom=raw_data['0'],
            color='#9a0941',
            edgecolor='white',
            width=barWidth, linewidth=0)
        plt.bar(
            r,
            raw_data['2'],
            bottom=[i+j for i, j in zip(raw_data['0'], raw_data['1'])],
            color='#b3a812',
            edgecolor='white',
            width=barWidth,
            linewidth=0)
        plt.bar(
            r,
            raw_data['3'],
            bottom=[i+j+k for i, j, k in zip(
                raw_data['0'], raw_data['1'], raw_data['2'])],
            color='#ed7d31',
            edgecolor='white',
            width=barWidth,
            linewidth=0)
        plt.bar(
            r,
            raw_data['4'],
            bottom=[i+j+k+l for i, j, k, l in zip(
                raw_data['0'], raw_data['1'], raw_data['2'], raw_data['3'])],
            color='#fed037',
            edgecolor='white',
            width=barWidth,
            linewidth=0)
        plt.bar(
            r,
            raw_data['ge5'],
            bottom=[i+j+k+l+m for i, j, k, l, m in zip(
                raw_data['0'], raw_data['1'], raw_data['2'],
                raw_data['3'], raw_data['4'])],
            color='#d35185',
            edgecolor='white',
            width=barWidth,
            linewidth=0)

        plt.xticks(r, names)
        plt.yticks(range(0, 110, 10))
        plt.xlabel("Maternal age at delivery")
        plt.ylabel("Percentage of pregnancies [%]")
        plt.title(('Anteil verschiedener bezogener Wirkstoffe während '
                   'der Schwangerschaft'))
        plt.title(('Percentage of claimed distinct drug prescriptions '
                   'during pregnancy'))

        plt.grid(axis='y', color='#888a88', linestyle=':', linewidth=0.5)
        plt.legend(
            ['0', '1', '2', '3', '4', '≥ 5'],
            loc='center left',
            bbox_to_anchor=(1.04, 0.5),
        )

        if save:
            fname_trunk = 'plot_C'
            if weighted:
                fname_trunk += '_weighted'
            fpath = os.path.join(path, fname_trunk)
            plt.savefig(fpath + '.png', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.png')
            plt.savefig(fpath + '.pdf')
            if log:
                console.log(f'Saved file {fpath}.pdf')
            plt.savefig(fpath + '.jpg', dpi=300)
            if log:
                console.log(f'Saved file {fpath}.jpg')
        rtime = time() - start
        if log:
            console.log(
                f'Generated plot C with weighted={weighted}'
                f' in {rtime:.1f} seconds')
        plt.close()


def _n_most_popular_codes(data, n=10):
    li = []
    for i in range(0, 5):
        li += list(data.sort_values(
            f'per_10000_T{i}', ascending=False)[:n]['atc_c'])
    return np.unique(li)


def _shorten_name(name):
    names = [
        'amoxicillin',
        'artificial tears',
        'pertussis',
        'influenza, inactivated']
    for x in names:
        if name.startswith(x):
            return x
    return name


def plot_popular_drugs(data, path, console, log=True, save=True):
    stext = '[bold green]Generating plot for popular prescriptions...'
    with console.status(stext):
        start = time()
        n_codes = _n_most_popular_codes(data, 16)
        data_columns = [
            'per_10000_T0',
            'per_10000_T1',
            'per_10000_T2',
            'per_10000_T3',
            'per_10000_T4']
        plot_data = data[
            data['atc_c'].isin(n_codes)][
                ['name'] + data_columns].fillna(0)
        x = np.arange(5)

        plt.rcParams['font.family'] = 'Arial'
        fig, subs = plt.subplots(6, 4, sharey=True, figsize=(8, 16))
        for row in range(6):
            for col in range(4):
                i = row * 6 + col
                values = plot_data.iloc[i][data_columns].values
                name = _shorten_name(plot_data.iloc[i]['name'])
                subs[row][col].plot(x, values, linestyle='-', marker='o')
                subs[row][col].set_title(name)
                subs[row][col].set_xticks(x, x)
        plt.tight_layout()

        if save:
            fname_trunk = 'plot_popular_drugs'
            fpath = os.path.join(path, fname_trunk)
            plt.savefig(fpath + '.pdf')
            if log:
                console.log(f'Saved file {fpath}.pdf')
        rtime = time() - start
        if log:
            console.log(
                'Generated plot for popular prescriptions '
                f'in {rtime:.1f} seconds')
        plt.close()


def plot_popular_drugs_single(data, path, console, n=100, log=True, save=True):
    stext = (f'[bold green]Generating single plots for the {n} most '
             'popular prescriptions...')
    with console.status(stext):
        start = time()
        data_columns = [
            'per_10000_T0',
            'per_10000_T1',
            'per_10000_T2',
            'per_10000_T3',
            'per_10000_T4']
        data = data.iloc[
            data[data_columns].sum(1).sort_values(
                ascending=False).index].fillna(0)
        x = np.arange(5)
        for i, (_, row) in enumerate(data.iterrows()):
            if i >= n:
                break
            name = row['name']
            atc_c = row['atc_c']
            values = row[data_columns].values
            plt.plot(x, values, linestyle='-', marker='o')
            plt.title(name + f' ({atc_c})')
            plt.xticks(x, x)
            plt.ylim(ymin=0)
            if save:
                fname_trunk = 'plot_' + str(i + 1).zfill(3) + '_popular_drugs'
                fpath = os.path.join(path, fname_trunk)
                plt.savefig(fpath + '.pdf')
                if log:
                    console.log(f'Saved file {fpath}.pdf')
            plt.close()
        rtime = time() - start
        if log:
            console.log(f'Generated single plots for the {n} most popular '
                        f'prescriptions in {rtime:.1f} seconds')
