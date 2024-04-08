import os
import pyreadstat
import pandas as pd
import numpy as np
from datetime import timedelta
from functools import reduce
from time import time

from tabulate import tabulate
from rich.console import Console
from statsmodels.stats.weightstats import DescrStatsW
from plots import (
    plot_a, plot_age_distribution, plot_c, plot_popular_drugs,
    plot_popular_drugs_single)


DATA_DIR = 'data'
RESULTS_DIR = 'results'
console = Console()


def read_data(datafile, log=True):
    with console.status(f'[bold green]Reading file {datafile}...'):
        start = time()
        if datafile.endswith('sas7bdat'):
            df, _ = pyreadstat.read_sas7bdat(
                datafile,
                dates_as_pandas_datetime=True,
                encoding='utf-8'
            )
        elif datafile.endswith('feather'):
            df = pd.read_feather(datafile)
        else:
            raise ValueError('Unknown filetype')
        if log:
            console.log(f'Dataframe is of shape {df.shape}.')
        rtime = time() - start
        if log:
            console.log(f'Read file {datafile} in {rtime:.1f} seconds')
        return df


def generate_sample(df, n_patientids=4000):
    patientids = df['patient'].unique()
    sample_ids = np.random.choice(patientids, size=n_patientids)
    return df[df['patient'].isin(sample_ids)]


def _save_csv_excel(fname_trunk, data, log=True):
    csv_fname = fname_trunk + '.csv'
    excel_fname = fname_trunk + '.xlsx'
    fpath_csv = os.path.join(RESULTS_DIR, csv_fname)
    fpath_excel = os.path.join(RESULTS_DIR, excel_fname)
    data.to_csv(fpath_csv)
    if log:
        console.log(f'Saved file {fpath_csv}')
    data.to_excel(fpath_excel)
    if log:
        console.log(f'Saved file {fpath_excel}')


def preprocess_pregpop_dataframe(df, log=True):
    with console.status('[bold green]Preprocessing pregpop dataframe...'):
        start = time()
        if log:
            console.log('Convertering `age_n`, `year` and `trimester` to data '
                        'type `int`')
        df['age_n'] = df['age_n'].astype('int')
        df['trimester'] = df['trimester'].astype('Int64')

        rem_columns = [
            'preg_seq',
            'ss_start_d',
            'ss_ende_d',
            'gender',
            'tarif_ziff_block',
            'wohn_kt_kbez_x',
            'bv_leist',
            'artikel_bez_x',
            'marke_artikel_bez_x',
            'lerb_og_x',
            'vlerb_og_x',
            'kanton',
            'hr_fk',
            'obs_end_d',
            'atc1',
            'Name_atc1',
            'atc3',
            'atc7',
            'Name_atc7',
            'hrf_b'
        ]
        if log:
            c = ', '.join(rem_columns)
            console.log(f'Removing columns {c}')
        df = df.drop(columns=rem_columns, errors='ignore')

        who_atc_fpath = os.path.join(DATA_DIR, 'WHO_ATC_2014.xlsx')
        if log:
            console.log(f'Obtaining names of ATC codes from {who_atc_fpath}')
        df_atc_bez = pd.read_excel(who_atc_fpath)
        df_atc_bez.rename(columns={'ATCCode': 'atc_c'}, inplace=True)
        df = pd.merge(df, df_atc_bez, on='atc_c', how='left')

        if log:
            console.log('Add column `atc4`')
        df['atc4'] = df['atc_c'].str[0:4]

        if log:
            console.log('Add column `quarter`')
        df['quarter'] = (((
                df['leist_datum'] - df['obs_beg_d'])
            ).dt.days.astype('Int64') // 90) + 1

        if log:
            c = ', '.join(df.columns)
            console.log(f'Columns of dataframe: {c}')
            console.log(f'Dataframe is of shape {df.shape}')

        if log:
            console.log('Changing type of column `year` to `Int64`')
        df['year'] = df['year'].astype('Int64')

        if log:
            console.log('Changing type of column `age_n` to `int`')
        df['age_n'] = df['age_n'].astype(int)

        rtime = time() - start
        if log:
            console.log(
                f'Preprocessed pregpop dataframe in {rtime:.1f} seconds')
        return df


def preprocess_childbear_dataframe(df, log=True):
    with console.status('[bold green]Preprocessing childbear dataframe...'):
        start = time()
        rem_columns = [
            'gender',
            'wohn_kt_kbez_x',
            'bv_leist',
            'artikel_bez_x',
            'marke_artikel_bez_x',
            'lerb_og_x',
            'vlerb_og_x',
            'kanton',
            'year',
            'atc1',
            'atc3',
            'atc7',
            'Name_atc1',
            'Name_atc3',
            'Name_atc7',
            'noclaim',
            'vers_start',
            'vers_end',
            'yearclaim'
        ]
        if log:
            c = ', '.join(rem_columns)
            console.log(f'Removing columns {c}')
        df = df.drop(columns=rem_columns, errors='ignore')

        if log:
            c = ', '.join(df.columns)
            console.log(f'Columns of dataframe: {c}')
            console.log(f'Dataframe is of shape {df.shape}')
        rtime = time() - start
        if log:
            console.log(
                f'Preprocessed childbear dataframe in {rtime:.1f} seconds')
        return df


def age_distribution(df, log=True, save=True):
    with console.status('[bold green]Calculating age distribution...'):
        df_age = df.groupby('idbirth')[
            ['age_n', 'preg_hrf_b']].aggregate(
                {'age_n': 'min', 'preg_hrf_b': 'first'})
        total = df_age.shape[0]
        total_sum_of_weights = df_age['preg_hrf_b'].sum()
        result = {}
        available_ages = np.unique(df_age['age_n'])
        for age in available_ages:
            isage = (df_age['age_n'] == age)
            quantity = df_age[isage].shape[0]
            sum_of_weights = df_age[isage]['preg_hrf_b'].sum()
            percent = quantity / total * 100
            percent_weighted = sum_of_weights / total_sum_of_weights * 100
            result[age] = {
                'quantity': quantity,
                'sum_of_weights': sum_of_weights,
                'percent': percent,
                'percent_weighted': percent_weighted
            }
        result = pd.DataFrame(result)
        if save:
            fpath = os.path.join(RESULTS_DIR, 'age_distribution.csv')
            result.to_csv(fpath)
            if log:
                console.log(f'Saved file {fpath}')
        return result


def _exclusions_overview_str(df, exclusions):
    out = df[exclusions][['atc_c', 'Name_atc3', 'Name']].value_counts()
    atc_c = [k[0] for k in out.keys()]
    types = [k[1] for k in out.keys()]
    names = [k[2] for k in out.keys()]
    values = out.values
    out_df = pd.DataFrame(
        {
            'atc_c': atc_c,
            'Type': types,
            'Name': names,
            'Number of occurences': values
        }
    )
    tab = tabulate(
        out_df,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        maxcolwidths=[8, 25, 40, 8],
    )
    return tab


def exclusions_a(df, log=True):
    with console.status('[bold green]Detecting exclusions A'):
        reg = r'''(?x)
                ^(?:
                    V(?!03)|
                    V03A(?:K|M|N)|
                    V03AB16|
                    V03AZ01|
                    B05|
                    A01AA|  # Caries prophylactic agents
                    A07CA|  # Oral rehydration salt formulations
                    D02|  # EMOLLIENTS AND PROTECTIVES
                    D09AX|  # Soft paraffin dressings
                    P03B|  # INSECTICIDES AND REPELLENTS
                    S01J|  # DIAGNOSTIC AGENTS
                    S01K|  # SURGICAL AIDS
                    S01XA20  # artificial tears and other indifferent prep.
                )\w*$'''
        exclusions = df['atc_c'].str.match(reg)
        if log:
            console.log('List of found exclusions A:')
            console.log(_exclusions_overview_str(df, exclusions))
        return exclusions


def exclusions_b(df, log=True):
    with console.status('[bold green]Detecting exclusions B'):
        reg = r'''(?x)
                ^(?:
                    A11|
                    A12|
                    B03A|
                    B03B|
                    H03C
                )\w*$
        '''
        exclusions = df['atc_c'].str.match(reg)
        if log:
            console.log('List of found exclusions B:')
            console.log(_exclusions_overview_str(df, exclusions))
        return exclusions


def exclusions_c(df, log=True):
    with console.status('[bold green]Detecting exclusions B'):
        excl_reg = r'''(?x)
                    ^(?:
                        G02B|
                        G03[ADG]|
                        D0[2-9]|
                        D11|
                        D0[15]A|
                        D10A|
                        [AG]01|
                        M02|
                        P03|
                        R0[24]|
                        S0[1-3]|
                        A1[1-2]|
                        J0[6-7]
                        C05[AB]|
                        N01B|
                        R01A|
                        B03[AB]|
                        H03C|
                        G02C[CD]|
                        H01C[AC]
                    )\w*$
        '''
        allow_reg = r'''(?x)
                        ^(?:
                            S01EC0[125]|
                            S01XA4[34]|
                            D02BB0[12]|
                            D02BB51|
                            D11AA0[49]|
                            D11AH0[45]|
                            D11AX02|
                            D11AX10
                        )\w*$
        '''
        excl = df['atc_c'].str.match(excl_reg)
        allow = df['atc_c'].str.match(allow_reg)
        exclusions = (~allow) & excl
        if log:
            console.log('List of found exclusions C:')
            console.log(_exclusions_overview_str(df, exclusions))
        return exclusions


def exclusions_d(df, log=True):
    with console.status('[bold green]Detecting exclusions B'):
        reg = r'''(?x)
                ^(?:
                    J07
                )\w*$
        '''
        exclusions = df['atc_c'].str.match(reg)
        if log:
            console.log('List of found exclusions D:')
            console.log(_exclusions_overview_str(df, exclusions))
        return exclusions


def _age_range(df, begin, end, key='age_n'):
    if not begin and not end:
        return pd.Series(True, df.index)
    return (df[key] >= begin) & (df[key] <= end)


def _prescriptions(df):
    return (df['atc_c'] != '') & (df['atc_c'] != 'z')


def _trimester(df, trim):
    if not trim or trim == 'any':
        return pd.Series(True, df.index)
    if trim == '123':
        return df['trimester'].isin([1, 2, 3])
    return df['trimester'] == int(trim)


def _year(df, year, key='year'):
    if not year or year == 'any':
        return pd.Series(True, df.index)
    return df[key] == year


def _calculate_stats(df, name, acol, wcol):
    quantity = df.shape[0]
    quantityw = df[wcol].sum()
    descr = df[acol].describe()
    mean = descr['mean']
    std = descr['std']
    min = descr['min']
    max = descr['max']
    q25 = descr['25%']
    q50 = descr['50%']
    q75 = descr['75%']

    wdf = DescrStatsW(df[acol], df[wcol])
    meanw = wdf.mean
    stdw = wdf.std
    q25w, q50w, q75w = wdf.quantile([0.25, 0.5, 0.75])
    return {
        name: {
            'quantity': quantity,
            'mean': mean,
            'std': std,
            'min': min,
            'max': max,
            'q25': q25,
            'q50 (median)': q50,
            'q75': q75,
            'quantity_weighted': quantityw,
            'mean_weighted': meanw,
            'std_weighted': stdw,
            'q25_weighted': q25w,
            'q50_weighted (median_weighted)': q50w,
            'q75_weighted': q75w
        }
    }


def study_population(df_pregpop, df_childbear, log=True, save=True):
    with console.status('[bold green]Analyzing study population...'):
        result = {}
        pregnancies = df_pregpop.groupby('idbirth')[[
            'idbirth', 'age_n', 'preg_hrf_b']].first()
        result.update(_calculate_stats(
            pregnancies, 'Pregnancies', 'age_n', 'preg_hrf_b'))
        for (age_s, age_e) in ((0, 25), (26, 35), (36, 100)):
            group = pregnancies[_age_range(pregnancies, age_s, age_e)]
            result.update(_calculate_stats(
                group, f'Pregnancies ({age_s} <= age <= {age_e})',
                'age_n', 'preg_hrf_b'))
        women_pregpop = df_pregpop.groupby(
            'patient')[['age_n', 'preg_hrf_b']].first()
        result.update(_calculate_stats(
            women_pregpop, 'Women pregpop', 'age_n', 'preg_hrf_b'))

        women_childbear = df_childbear.groupby('patient')[[
            'patient', 'age', 'hrf_b']].first()
        result.update(_calculate_stats(
            women_childbear, 'Women childbear', 'age', 'hrf_b'))
        for (age_s, age_e) in ((0, 25), (26, 35), (36, 100)):
            group = women_childbear[_age_range(
                women_childbear, age_s, age_e, key='age')]
            result.update(_calculate_stats(
                women_childbear,
                f'Women childbear ({age_s} <= age <= {age_e})',
                'age', 'hrf_b'))

        if log:
            console.log('Analyzed study population')
        result = pd.DataFrame(result)
        if save:
            _save_csv_excel('study_population', result)
        return result


def count_medication_within_time_periods(
        df, excl_a=None, excl_b=None, excl_c=None, excl_d=None,
        log=True, save=True):
    with console.status(
            '[bold green]Counting medication within time periods...'):
        start = time()
        df['n_meds'] = 1 * _prescriptions(df)
        excl_a = exclusions_a(df, log=False) if excl_a is None else excl_a
        excl_b = exclusions_b(df, log=False) if excl_b is None else excl_b
        excl_c = exclusions_c(df, log=False) if excl_c is None else excl_c
        excl_d = exclusions_d(df, log=False) if excl_d is None else excl_d
        no_excl = pd.Series(False, df.index)

        names = [
            'No Exclusions',
            'Exclude A',
            'Exclude A and B',
            'Exclude A, B and D'
        ]
        allows = [
            ~no_excl,
            ~excl_a,
            ~excl_a & ~excl_b,
            ~excl_a & ~excl_b & ~excl_d
        ]
        data = {
            'Name': [],
            'T0': [],
            'T0_weighted': [],
            'T0_total': [],
            'T0_total_weighted': [],
            'T1': [],
            'T1_weighted': [],
            'T1_total': [],
            'T1_total_weighted': [],
            'T2': [],
            'T2_weighted': [],
            'T2_total': [],
            'T2_total_weighted': [],
            'T3': [],
            'T3_weighted': [],
            'T3_total': [],
            'T3_total_weighted': [],
            'T4': [],
            'T4_weighted': [],
            'T4_total': [],
            'T4_total_weighted': [],
            'T123': [],
            'T123_weighted': [],
            'T123_total': [],
            'T123_total_weighted': []
        }
        for (name, allow) in zip(names, allows):
            data['Name'].append(name)
            for trim in range(0, 5):
                dft = df[allow & _trimester(df, trim)][
                    ['n_meds', 'preg_hrf_b']]
                data[f'T{trim}'].append(dft['n_meds'].sum())
                data[f'T{trim}_weighted'].append((
                    dft['n_meds'] * dft['preg_hrf_b']).sum())
                data[f'T{trim}_total'].append(dft['n_meds'].shape[0])
                data[f'T{trim}_total_weighted'].append(dft['preg_hrf_b'].sum())
            tr = df['trimester'].isin([1, 2, 3])
            dft = df[allow & tr][['n_meds', 'preg_hrf_b']]
            data['T123'].append(dft['n_meds'].sum())
            data['T123_weighted'].append((
                dft['n_meds'] * dft['preg_hrf_b']).sum())
            data['T123_total'].append(dft['n_meds'].shape[0])
            data['T123_total_weighted'].append(dft['preg_hrf_b'].sum())
        df_result = pd.DataFrame(data)
        if save:
            _save_csv_excel('total_medication', df_result)
        rtime = time() - start
        if log:
            console.log(('Counted medication within time periods '
                         f'in {rtime:.1f} seconds'))
        return df_result


def _medication_distribution(
        df, exclusions, trimester, age_s, age_e, nbins=6):
    selection = df[
        ~exclusions & _trimester(df, trimester)
        & _age_range(df, age_s, age_e)][['idbirth', 'atc_c']]

    # All idbirths that lie within the desired age range:
    idbirths = df[_age_range(df, age_s, age_e)]['idbirth'].unique()

    # Add these idbirths to the selection with an empty `atc_c` code,
    # so that pregnencies without a prescription during that time period
    # are also considered.
    add = pd.DataFrame({
        'idbirth': idbirths,
        'atc_c': ['' for _ in idbirths]
    })
    work_df = pd.concat([selection, add])

    meds = work_df.groupby('idbirth')['atc_c'].apply(list)
    nmeds = meds.apply(lambda x: len([x for x in np.unique(x) if x != '']))
    nmeds.rename('n_meds', inplace=True)
    weights = df.groupby('idbirth')['preg_hrf_b'].first()
    nmeds_df = pd.merge(nmeds, weights, on='idbirth')
    nmeds_series = [nmeds_df['n_meds'] == i for i in range(0, nbins - 1)]
    nmeds_series.append(nmeds_df['n_meds'] >= nbins - 1)

    total = nmeds_df.shape[0]
    total_weighted = nmeds_df['preg_hrf_b'].sum()
    n = [sum(x) for x in nmeds_series]
    n_weighted = [sum(nmeds_df[x]['preg_hrf_b']) for x in nmeds_series]
    perc = [x / total * 100.0 for x in n]
    weighted_perc = [
        sum(nmeds_df[x]['preg_hrf_b']) /
        sum(nmeds_df['preg_hrf_b']) * 100.0
        for x in nmeds_series]
    return total, total_weighted, n, n_weighted, perc, weighted_perc


def medication_distribution(df, exclusions, idapx, nbins=6):
    if nbins < 2:
        raise ValueError('nbins must be at least 2')
    distributions = {}
    for agegroup in ((0, 25), (26, 35), (36, 100), (0, 100)):
        for trim in ('0', '1', '2', '3', '4', '123'):
            basename = f'T{trim}_A{agegroup[0]}-{agegroup[1]}_{idapx}'
            (
                total, total_weighted, n, n_weighted, perc, weighted_perc
            ) = _medication_distribution(
                df, exclusions, trim,
                agegroup[0], agegroup[1], nbins=nbins
            )
            distributions.update({
                basename + '_total': total,
                basename + '_total_weighted': total_weighted,
                basename: n,
                basename + '_weighted': n_weighted,
                basename + '_percent': perc,
                basename + '_percent_weighted': weighted_perc
            })
    return distributions


def all_medication_distribution(
        df, excl_a, excl_b, excl_d, nbins=6, log=True, save=True):
    distributions = {}

    idapx = 'excl_a'
    process_text = f'medication distribution with nbins={nbins} for {idapx}'
    with console.status('[bold green]Calculating ' + process_text + '...'):
        start = time()
        distributions.update(
            medication_distribution(
                df, excl_a, idapx, nbins=nbins
            )
        )
        rtime = time() - start
        if log:
            console.log(
                'Calculated ' + process_text + f' in {rtime:.1f} seconds')

    idapx = 'excl_a+b+d'
    with console.status('[bold green]Calculating ' + process_text + '...'):
        start = time()
        distributions.update(
            medication_distribution(
                df, excl_a | excl_b | excl_d,
                idapx, nbins=nbins
            )
        )
        rtime = time() - start
        if log:
            console.log(
                'Calculated ' + process_text + f' in {rtime:.1f} seconds')

    frame = pd.DataFrame(distributions)
    frame.insert(
        0, column='Anzahl',
        value=[str(x) for x in range(0, nbins - 1)] + [f'>={nbins - 1}'])
    if save:
        _save_csv_excel(f'medication_distribution_nbins={nbins}', frame)
    return frame


def popular_drugs(df, exclusions, trimester, age_s, age_e, n='all'):
    selection = df[
        ~exclusions &
        _trimester(df, trimester) &
        _age_range(df, age_s, age_e) &
        ~df['atc_c'].isin(['', 'z'])][
            ['idbirth', 'atc_c', 'Name', 'preg_hrf_b']]

    # Find the most popular drugs:
    most_popular = selection[['atc_c', 'Name']].value_counts()
    if n != 'all':
        most_popular = most_popular[:n]

    prescr_df = df[
        ~exclusions &
        _trimester(df, trimester) &
        _age_range(df, age_s, age_e) &
        ~df['atc_c'].isin(['', 'z'])
    ][['idbirth', 'atc_c', 'preg_hrf_b']]
    npresc = prescr_df.shape[0]
    npresc_weighted = prescr_df['preg_hrf_b'].sum()

    result = {}
    result['atc_c'] = [x[0] for x in most_popular.index]
    result['name'] = [x[1] for x in most_popular.index]
    result['quantity'] = most_popular.values
    result['percent'] = result['quantity'] / npresc * 100
    result['per_10000'] = result['quantity'] / npresc * 10_000

    df_mpop = selection[selection['atc_c'].isin(result['atc_c'])]

    result['quantity_weighted'] = []
    result['percent_weighted'] = []
    result['per_10000_weighted'] = []

    for atc_c in result['atc_c']:
        s = df_mpop[df_mpop['atc_c'] == atc_c]['preg_hrf_b'].sum()
        result['quantity_weighted'].append(s)
        result['percent_weighted'].append(s / npresc_weighted * 100)
        result['per_10000_weighted'].append(
            s / npresc_weighted * 10_000)
    return result


def popular_drugs_per_pregnancy_all(df, exclusions, save=True):
    for (age_s, age_e) in ((0, 25), (26, 35), (36, 100), (0, 100)):
        for trimester in ('0', '1', '2', '3', '4', '123'):
            popular_drugs_per_pregnancy(
                    df, exclusions, trimester, age_s, age_e, save=save)


def popular_drugs_per_pregnancy(df, exclusions, trimester,
                                age_s, age_e, save=True):
    with console.status('[bold green]Analyzing popular drugs per pregnancy for'
                        f' trimester {trimester}, age start {age_s}, '
                        f'age end {age_e}'):
        selection = df[
            ~exclusions &
            _trimester(df, trimester) &
            _age_range(df, age_s, age_e) &
            ~df['atc_c'].isin(['', 'z'])][
                ['idbirth', 'atc_c', 'Name', 'preg_hrf_b']]

        # Add these idbirths to the selection with an empty `atc_c` code,
        # so that pregnencies without a prescription during that time period
        # are also considered.
        # add = pd.DataFrame({
        #     'idbirth': idbirths,
        #     'atc_c': ['' for _ in idbirths],
        # })
        add = pd.DataFrame({
            'idbirth': df[_age_range(df, age_s, age_e)]['idbirth'],
            'atc_c': ['' for _ in df[_age_range(df, age_s, age_e)]['idbirth']],
            'preg_hrf_b': df[_age_range(df, age_s, age_e)]['preg_hrf_b'],
        })
        selection = pd.concat([selection, add])

        most_popular = selection[['atc_c', 'Name']].value_counts()
        most_popular = most_popular[most_popular >= 100]
        codes = [x[0] for x in most_popular.keys()]
        names = [x[1] for x in most_popular.keys()]
        new_columns = []
        for code in codes:
            nc = (selection['atc_c'] == code)
            nc.rename(code, inplace=True)
            new_columns.append(nc)
        ncdf = pd.concat(new_columns, axis=1)
        df = pd.concat([selection, ncdf], axis=1)
        dfidb = df.groupby('idbirth')[['preg_hrf_b'] + codes].max()

        result = {
                'ATC_C': codes,
                'Name': names,
                'N Total': [],
                'N': [],
                'Perc Preg': [],
                'N Total Weighted': [],
                'N Weighted': [],
                'Perc Weighted': [],
        }

        for code in codes:
            n_positive = dfidb[code].sum()
            perc = n_positive / dfidb.shape[0] * 100
            result['N'].append(n_positive)
            result['N Total'].append(dfidb.shape[0])
            result['Perc Preg'].append(perc)
            n_positive_weighted = dfidb[dfidb[code]]['preg_hrf_b'].sum()
            result['N Weighted'].append(n_positive_weighted)
            result['N Total Weighted'].append(dfidb['preg_hrf_b'].sum())
            result['Perc Weighted'].append(
                    n_positive_weighted / dfidb['preg_hrf_b'].sum() * 100)

        df_result = pd.DataFrame(result)
        df_result = df_result.sort_values(by=['Perc Preg'], ascending=False)
        if save:
            _save_csv_excel(
                    (f'popdrugs_per_preg_trimester={trimester}'
                     '_age_s={age_s}_age_e={age_e}'), df_result)


def all_popular_drugs(df, exclusions, idapx, log=True, save=True):
    results = {}
    with console.status('[bold green]Analyzing popular drugs...') as status:
        start = time()
        for (age_s, age_e) in ((0, 25), (26, 35), (36, 100), (0, 100)):
            for trimester in ('0', '1', '2', '3', '4', '123'):
                text = (f'popular drugs for age_s={age_s}, age_e={age_e}, '
                        f'trimester={trimester}, idapx={idapx}')
                status.update('[bold green]Finding ' + text + '...')
                result = popular_drugs(df, exclusions, trimester, age_s, age_e)
                result_df = pd.DataFrame(result)
                selectinfo = f'T{trimester}_A{age_s}-{age_e}_{idapx}'
                results.update(
                    {
                        selectinfo: result_df
                    }
                )
                if log:
                    console.log('Found ' + text)
                if save:
                    fname_trunk = f'popular_drugs_{selectinfo}'
                    _save_csv_excel(fname_trunk, result_df)
        rtime = time() - start
        if log:
            console.log(f'Analyzed popular drugs in {rtime:.1f} seconds')

        return results


def _merge_popular_drugs_results(data, age_s, age_e, idapx):
    d0 = data[f'T0_A{age_s}-{age_e}_{idapx}']
    d1 = data[f'T1_A{age_s}-{age_e}_{idapx}']
    d2 = data[f'T2_A{age_s}-{age_e}_{idapx}']
    d3 = data[f'T3_A{age_s}-{age_e}_{idapx}']
    d4 = data[f'T4_A{age_s}-{age_e}_{idapx}']
    dlist = [d0, d1, d2, d3, d4]
    dlist2 = []
    for i, d in enumerate(dlist):
        new = d.rename(
            columns={
                'quantity': f'quantity_T{i}',
                'quantity_weighted': f'quantity_weighted_T{i}',
                'percent': f'percent_T{i}',
                'per_10000': f'per_10000_T{i}',
                'percent_weighted': f'percent_weighted_T{i}',
                'per_10000_weighted': f'per_10000_weighted_T{i}'
            },
        )
        dlist2.append(new)
    return reduce(
        lambda x, y: pd.merge(x, y, on=['atc_c', 'name'], how='outer'), dlist2)


def multiple_prescriptions(
        df, exclusions, atc_column_name, log=True, save=True):
    stext = '[bold green]Analyzing multiple prescriptions...'
    with console.status(stext) as status:
        selection0 = df[
            ~exclusions &
            _trimester(df, '0') &
            (~df[atc_column_name].isin(['', 'z']))].copy()
        selection0['n'] = 1
        selection123 = df[
            ~exclusions &
            _trimester(df, '123') &
            (~df[atc_column_name].isin(['', 'z']))].copy()
        selection123['n'] = 1
        selection4 = df[
            ~exclusions &
            _trimester(df, '0') &
            (~df[atc_column_name].isin(['', 'z']))].copy()
        selection4['n'] = 1

        tmp0 = selection0.groupby(
            ['idbirth', atc_column_name])['n'].sum()
        tmp123 = selection123.groupby(
            ['idbirth', atc_column_name])['n'].sum()
        tmp4 = selection0.groupby(
            ['idbirth', atc_column_name])['n'].sum()

        # Two or more occurences of the same code before pregnancy
        start = time()
        text = 'multiple prescriptions: Two or more before pregnancy'
        status.update('[bold green]Analyzing ' + text + '...')
        two_or_more_trim0 = tmp0.groupby('idbirth').max() >= 2
        two_or_more_trim0.rename('two_or_more_trim0', inplace=True)

        # Continuation during/after pregnancy:
        two_or_more_trim0_cont123 = (
            pd.merge(
                tmp0, tmp123,
                on=['idbirth', atc_column_name]
            )['n_x'] >= 2).groupby('idbirth').max()
        two_or_more_trim0_cont123.rename(
            'two_or_more_trim0_cont123', inplace=True)
        two_or_more_trim0_cont4 = (
            pd.merge(
                tmp0, tmp4,
                on=['idbirth', atc_column_name]
            )['n_x'] >= 2).groupby('idbirth').max()
        two_or_more_trim0_cont4.rename(
            'two_or_more_trim0_cont4', inplace=True)
        rtime = time() - start
        if log:
            console.log('Analyzed ' + text + f' in {rtime:.1f} seconds')

        # Two or more occurences of the same code before pregnancy
        # in different quarters:
        start = time()
        text = ('multiple prescriptions: Two or more before pregnancy '
                'in different quarters')
        status.update('[bold green]Analyzing ' + text + '...')
        tmp = selection0.groupby(
            ['idbirth', atc_column_name])['quarter'].apply('nunique')
        two_or_more_trim0_diff_quarters = tmp.groupby('idbirth').max() >= 2
        two_or_more_trim0_diff_quarters.rename(
            'two_or_more_trim0_diff_quarters',
            inplace=True)

        # Continuation during/after pregnancy:
        two_or_more_trim0_diff_quarters_cont123 = (pd.merge(
            tmp, tmp123,
            on=['idbirth', atc_column_name]
            )['quarter'] >= 2).groupby('idbirth').max()
        two_or_more_trim0_diff_quarters_cont123.rename(
            'two_or_more_trim0_diff_quarters_cont123',
            inplace=True)
        two_or_more_trim0_diff_quarters_cont4 = (pd.merge(
            tmp, tmp4,
            on=['idbirth', atc_column_name]
            )['quarter'] >= 2).groupby('idbirth').max()
        two_or_more_trim0_diff_quarters_cont4.rename(
            'two_or_more_trim0_diff_quarters_cont4',
            inplace=True)
        rtime = time() - start
        if log:
            console.log('Analyzed ' + text + f' in {rtime:.1f} seconds')

        # Two or more occurences of the same code before pregnancy
        # in different quarters and reference dates more than 14
        # days apart:
        start = time()
        text = ('multiple prescriptions: Two or more before pregnancy in '
                'different quarters and at least 14 days apart')
        status.update('[bold green]Analyzing ' + text + '...')
        tmp = selection0.groupby(
            ['idbirth', atc_column_name])[
                ['quarter', 'leist_datum', 'n']].aggregate(
                    {
                        'quarter': 'nunique',
                        'leist_datum': np.ptp,
                        'n': 'sum'
                    })
        tmp_save = tmp.copy()
        tmp['x'] = (tmp['quarter'] >= 2) & (
            tmp['leist_datum'] >= timedelta(days=14))
        two_or_more_trim0_diff_quarters_mt_14_days = tmp.groupby(
            'idbirth')['x'].max()
        two_or_more_trim0_diff_quarters_mt_14_days.rename(
            'two_or_more_trim0_diff_quarters_mt_14_days',
            inplace=True)

        # Continuation during/after pregnancy
        two_or_more_trim0_diff_quarters_mt_14_days_cont123 = pd.merge(
            tmp, tmp123,
            on=['idbirth', atc_column_name]).groupby('idbirth')['x'].max()
        two_or_more_trim0_diff_quarters_mt_14_days_cont123.rename(
            'two_or_more_trim0_diff_quarters_mt_14_days_cont123',
            inplace=True)

        two_or_more_trim0_diff_quarters_mt_14_days_cont4 = pd.merge(
            tmp, tmp4,
            on=['idbirth', atc_column_name]).groupby('idbirth')['x'].max()
        two_or_more_trim0_diff_quarters_mt_14_days_cont4.rename(
            'two_or_more_trim0_diff_quarters_mt_14_days_cont4',
            inplace=True)
        rtime = time() - start
        if log:
            console.log('Analyzed ' + text + f' in {rtime:.1f} seconds')

        # Three or more occurences of the same code before pregnancy
        # in at least two different quarters and reference dates more than 14
        # days apart:
        start = time()
        text = ('multiple prescriptions: Three or more before pregnancy in '
                'at least two different quarters and at least 14 days apart')
        status.update('[bold green]Analyzing ' + text + '...')
        tmp = tmp_save.copy()
        tmp['x'] = (tmp['n'] >= 3) & (
            tmp['leist_datum'] >= timedelta(days=14)) & (
            tmp['quarter'] >= 2)
        three_or_more_trim0_diff_quarters_mt_14_days = tmp.groupby(
            'idbirth')['x'].max()
        three_or_more_trim0_diff_quarters_mt_14_days.rename(
            'three_or_more_trim0_diff_quarters_mt_14_days',
            inplace=True)

        # Continuation during/after pregnancy:
        three_or_more_trim0_diff_quarters_mt_14_days_cont123 = pd.merge(
            tmp, tmp123,
            on=['idbirth', atc_column_name]).groupby('idbirth')['x'].max()
        three_or_more_trim0_diff_quarters_mt_14_days_cont123.rename(
            'three_or_more_trim0_diff_quarters_mt_14_days_cont123',
            inplace=True)
        three_or_more_trim0_diff_quarters_mt_14_days_cont4 = pd.merge(
            tmp, tmp4,
            on=['idbirth', atc_column_name]).groupby('idbirth')['x'].max()
        three_or_more_trim0_diff_quarters_mt_14_days_cont4.rename(
            'three_or_more_trim0_diff_quarters_mt_14_days_cont4',
            inplace=True)
        rtime = time() - start
        if log:
            console.log('Analyzed ' + text + f' in {rtime:.1f} seconds')

        # One or more occurences of the same code before pregnancy:
        start = time()
        text = 'multiple prescriptions: One or more before pregnancy'
        status.update('[bold green]Analyzing ' + text + '...')
        tmp = selection0.groupby(['idbirth', atc_column_name])['n'].sum() >= 1
        one_or_more_trim0 = tmp.groupby('idbirth').max()
        one_or_more_trim0.rename('one_or_more_trim0', inplace=True)

        # Continuation during/after pregnancy:
        one_or_more_trim0_cont123 = pd.merge(
            tmp, tmp123,
            on=['idbirth', atc_column_name]).groupby('idbirth')['n_x'].max()
        one_or_more_trim0_cont123.rename(
            'one_or_more_trim0_cont123', inplace=True)
        one_or_more_trim0_cont4 = pd.merge(
            tmp, tmp4,
            on=['idbirth', atc_column_name]).groupby('idbirth')['n_x'].max()
        one_or_more_trim0_cont4.rename('one_or_more_trim0_cont4', inplace=True)
        rtime = time() - start
        if log:
            console.log('Analyzed ' + text + f' in {rtime:.1f} seconds')

        status.update(
            '[bold green] Combining the results and calculating the ratios...')
        idbirths = pd.DataFrame({'idbirth': df['idbirth'].unique()})
        weights = df[['idbirth', 'preg_hrf_b']].groupby('idbirth').first()
        overview = pd.merge(idbirths, weights, on='idbirth')
        items = (
            two_or_more_trim0,
            two_or_more_trim0_cont123,
            two_or_more_trim0_cont4,
            two_or_more_trim0_diff_quarters,
            two_or_more_trim0_diff_quarters_cont123,
            two_or_more_trim0_diff_quarters_cont4,
            two_or_more_trim0_diff_quarters_mt_14_days,
            two_or_more_trim0_diff_quarters_mt_14_days_cont123,
            two_or_more_trim0_diff_quarters_mt_14_days_cont4,
            three_or_more_trim0_diff_quarters_mt_14_days,
            three_or_more_trim0_diff_quarters_mt_14_days_cont123,
            three_or_more_trim0_diff_quarters_mt_14_days_cont4,
            one_or_more_trim0,
            one_or_more_trim0_cont123,
            one_or_more_trim0_cont4
        )
        for item in items:
            overview = pd.merge(overview, item, how='left', on='idbirth')
        overview.fillna(False, inplace=True)

        results = {}
        for item in items:
            total = overview.shape[0]
            total_weighted = overview['preg_hrf_b'].sum()
            quantity = overview[item.name].sum()
            quantity_weighted = (
                overview[item.name] * overview['preg_hrf_b']).sum()
            percent = quantity / total * 100
            per_10000 = quantity / total * 10_000
            percent_weighted = quantity_weighted / (
                    total_weighted) * 100
            per_10000_weighted = quantity_weighted / (
                    total_weighted) * 10_000
            results.update({
                item.name: {
                    'total': total,
                    'total_weighted': total_weighted,
                    'quantity': quantity,
                    'quantity_weighted': quantity_weighted,
                    'percent': percent,
                    'per_10000': per_10000,
                    'percent_weighted': percent_weighted,
                    'per_10000_weighted': per_10000_weighted
                }
            })
        df_results = pd.DataFrame(results)
        if save:
            _save_csv_excel('multiple_prescriptions', df_results)
        return results


def multiple_prescriptions_per_atc4(df, log=True, save=True):
    stext = '[bold green]Analyzing multiple prescriptions per atc4'
    with console.status(stext + '...') as status:
        selection0 = df[_trimester(df, '0') & (df['atc4'] != '')].copy()
        selection123 = df[_trimester(df, '123') & (df['atc4'] != '')].copy()
        selection4 = df[_trimester(df, '4') & (df['atc4'] != '')].copy()

        base = df.groupby('idbirth')[['idbirth', 'preg_hrf_b']].first()
        total = base.shape[0]
        totalw = base['preg_hrf_b'].sum()

        atc_codes = np.unique(selection0['atc4'])
        result = {}
        for code in atc_codes:
            status.update(stext + f' for code {code}...')
            s0 = selection0[selection0['atc4'] == code].copy()
            s123 = selection123[selection123['atc4'] == code].copy()
            s4 = selection4[selection4['atc4'] == code].copy()
            s0['n'] = 1
            s123['x123'] = True
            s4['x4'] = True
            res = s0.groupby('idbirth')[[
                'quarter', 'leist_datum', 'n', 'preg_hrf_b']].aggregate(
                    {
                        'quarter': 'nunique',
                        'leist_datum': np.ptp,
                        'n': 'sum',
                        'preg_hrf_b': 'first'
                    }
                )
            res['x0'] = (res['quarter'] >= 2) & (
                res['leist_datum'] >= timedelta(days=14)) & (res['n'] >= 2)
            tmp = s123.groupby('idbirth')['x123'].max()
            tmp.rename('x123', inplace=True)
            res = pd.merge(
                left=res, right=tmp, on='idbirth', how='left')[[
                    'preg_hrf_b', 'x0', 'x123']]
            tmp = s4.groupby('idbirth')['x4'].max()
            tmp.rename('x4', inplace=True)
            res = pd.merge(
                left=res, right=tmp, on='idbirth', how='left').fillna(False)
            quantity0 = res['x0'].sum()
            quantity123 = (res['x0'] & res['x123']).sum()
            quantity4 = (res['x0'] & res['x4']).sum()
            quantity0w = (res['x0'] * res['preg_hrf_b']).sum()
            quantity123w = (
                (res['x0'] & res['x123']) * res['preg_hrf_b']).sum()
            quantity4w = ((res['x0'] & res['x4']) * res['preg_hrf_b']).sum()
            percent0 = quantity0 / total * 100
            percent0w = quantity0w / totalw * 100
            percent123 = (
                quantity123 / quantity0 * 100 if quantity0 != 0 else 0.0)
            percent123w = quantity123w / quantity0w * 100 if abs(
                quantity0w) > 1e-4 else 0.0
            percent4 = quantity4 / quantity0 * 100 if quantity0 != 0 else 0.0
            percent4w = quantity4w / quantity0w * 100 if abs(
                quantity0w) > 1e-4 else 0.0
            per_10000_0 = quantity0 / total * 10_000
            per_10000_0w = quantity0w / totalw * 10_000
            result.update({
                code: {
                    'quantity0': quantity0,
                    'quantity0_weighted': quantity0w,
                    'percent0': percent0,
                    'percent0_weighted': percent0w,
                    'per_10000_0': per_10000_0,
                    'per_10000_0_weighted': per_10000_0w,
                    'quantity123': quantity123,
                    'quantity123_weighted': quantity123w,
                    'percent123 wrt. quantity0': percent123,
                    'percent123_weighted wrt. quantity0_weighted': percent123w,
                    'quantity4': quantity4,
                    'quantity4_weighted': quantity4w,
                    'percent4 wrt. quantity0': percent4,
                    'percent4_weighted wrt. quantity0_weighted': percent4w,
                    'total': total,
                    'total_weighted': totalw
                }
            })
        result_df = pd.DataFrame.from_dict(
            result, orient='index').sort_values('quantity0', ascending=False)
        if save:
            _save_csv_excel('multiple_prescriptions_per_atc4', result_df)
        if log:
            console.log('Analyzed multiple prescriptions per atc4')
        return result_df


def _list_to_regex(li):
    content = '|'.join(li)
    return rf'^(?:{content})\w*$'


def med_stats_per_pregnancy_all(df, excl_a, excl_b, excl_d, save=True):
    exclusions = [excl_a, excl_a | excl_b | excl_d]
    excl_strs = ['excl_a', 'excl_a+b+d']

    result = {}

    for unique_meds in (False, True):
        for (excl, excl_str) in zip(exclusions, excl_strs):
            for (age_s, age_e) in ((0, 25), (26, 35), (36, 100), (0, 100)):
                for trimester in ('0', '1', '2', '3', '4', '123'):
                    stats = med_stats_per_pregnancy(
                            df, excl, trimester, age_s, age_e,
                            excl_str, unique_meds=unique_meds)
                    result.update(stats)

    result = pd.DataFrame(result)
    if save:
        _save_csv_excel('med_stats', result)
    return result


def med_stats_per_pregnancy(df, exclusions, trimester, age_s, age_e,
                            excl_str, unique_meds=False):
    df['nmeds'] = ~df['atc_c'].isin(('z', ''))

    selection = df[
            ~exclusions &
            _trimester(df, trimester) &
            _age_range(df, age_s, age_e)
            ][['idbirth', 'atc_c', 'preg_hrf_b', 'nmeds']]

    # Add these idbirths to the selection with an empty `atc_c` code,
    # so that pregnencies without a prescription during that time period
    # are also considered.
    # add = pd.DataFrame({
    #     'idbirth': idbirths,
    #     'atc_c': ['' for _ in idbirths],
    # })
    add = pd.DataFrame({
        'idbirth': df[_age_range(df, age_s, age_e)]['idbirth'],
        'atc_c': ['' for _ in df[_age_range(df, age_s, age_e)]['idbirth']],
        'preg_hrf_b': df[_age_range(df, age_s, age_e)]['preg_hrf_b'],
        'nmeds': [False for _ in df[_age_range(df, age_s, age_e)]['idbirth']],
    })
    selection = pd.concat([selection, add])

    if unique_meds:
        selection = selection.drop_duplicates(subset=['idbirth', 'atc_c'])

    name = (f'Med Stats {excl_str} trimester={trimester} age_s={age_s} '
            'age_e={age_e} unique_meds={unique_meds}')
    nmeds_total = selection.groupby(
            'idbirth')[['nmeds', 'preg_hrf_b']].aggregate(
                    {'nmeds': np.sum, 'preg_hrf_b': 'first'})
    return _calculate_stats(nmeds_total, name, 'nmeds', 'preg_hrf_b')


def run_all():
    start = time()

    fname_pregpop = 'preg_claims_noabort_2.feather'
    df_pregpop = read_data(os.path.join(DATA_DIR, fname_pregpop))
    df_pregpop = preprocess_pregpop_dataframe(df_pregpop)
    fname_childbear = 'childbear21_2.feather'
    df_childbear = read_data(os.path.join(DATA_DIR, fname_childbear))
    df_childbear = preprocess_childbear_dataframe(df_childbear)
    excl_a = exclusions_a(df_pregpop)
    excl_b = exclusions_b(df_pregpop)
    excl_c = exclusions_c(df_pregpop)
    excl_d = exclusions_d(df_pregpop)

    med_stats_per_pregnancy_all(df_pregpop, excl_a, excl_b, excl_d)

    popular_drugs_per_pregnancy_all(df_pregpop, excl_a | excl_b | excl_d)

    study_population(df_pregpop, df_childbear)

    df_age_distribution = age_distribution(df_pregpop)
    plot_age_distribution(df_age_distribution, RESULTS_DIR, console)
    plot_age_distribution(
        df_age_distribution, RESULTS_DIR, console, weighted=True)

    count_medication_within_time_periods(
        df_pregpop, excl_a, excl_b, excl_c, excl_d)

    df_medication_distribution_0_1 = all_medication_distribution(
        df_pregpop, excl_a, excl_b, excl_d, nbins=2
    )
    plot_a(df_medication_distribution_0_1, RESULTS_DIR, console)
    plot_a(df_medication_distribution_0_1, RESULTS_DIR, console, weighted=True)

    df_medication_distribution = all_medication_distribution(
        df_pregpop, excl_a, excl_b, excl_d)
    plot_c(df_medication_distribution, RESULTS_DIR, console)
    plot_c(df_medication_distribution, RESULTS_DIR, console, weighted=True)

    popular_drugs_results = all_popular_drugs(
        df_pregpop, excl_a | excl_b | excl_d, 'excl_a+b+d')
    popular_drugs_results = all_popular_drugs(
        df_pregpop, excl_a | excl_b | excl_d, 'excl_a+b+d')

    merged_popular_drugs_results = _merge_popular_drugs_results(
        popular_drugs_results, 0, 100, 'excl_a+b+d')
    merged_popular_drugs_results = _merge_popular_drugs_results(
        popular_drugs_results, 0, 100, 'excl_a+b+d')

    plot_popular_drugs(merged_popular_drugs_results, RESULTS_DIR, console)
    plot_popular_drugs_single(
        merged_popular_drugs_results, RESULTS_DIR, console)

    multiple_prescriptions(
        df_pregpop, excl_a | excl_b | excl_c | excl_d, 'atc4')

    multiple_prescriptions_per_atc4(df_pregpop)

    rtime = time() - start
    console.log(f'Analysis done! Total runtime: {rtime:.1f} seconds')


if __name__ == '__main__':
    run_all()
