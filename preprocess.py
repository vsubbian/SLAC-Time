from util import *
import numpy as np
import pandas as pd


DATA_PATH = 'data/'
np.random.seed(7)


class Connection:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        self.clinic_data = pd.read_csv(DATA_PATH + 'NSF U01 Clinical Data.csv', skipinitialspace=True)
        self.gcs_data = pd.read_csv(DATA_PATH + 'NSF U01 GCS and Pupillary Data.csv', skipinitialspace=True)
        self.lab_data = pd.read_csv(DATA_PATH + 'NSF U01 Lab Data.csv', skipinitialspace=True)
        self.vital_data = pd.read_csv(DATA_PATH + 'NSF U01 Vitals Data.csv', skipinitialspace=True)
        self.included_cases = pd.read_csv(DATA_PATH + 'Included Subjects.csv', skipinitialspace=True)
        self.cleaned_version = False
        self.all_clean()

    def all_clean(self):
        na_dict = {'Unknown': np.nan,
                   'Unknown ': np.nan,
                   'Unknown  ': np.nan,
                   'Invalid': np.nan,
                   'UNK/Untestable': np.nan,
                   '': np.nan}
        self.clinic_data.replace(na_dict, inplace=True)
        self.vital_data.replace(na_dict, inplace=True)
        self.lab_data.replace(na_dict, inplace=True)
        self.gcs_data.replace(na_dict, inplace=True)
    
    def clean_clinic_data(self, miss_rate=0.3, cleaned_version=False, max_gcs=15,
                          ext_subjects=True):
        self.cleaned_version = cleaned_version

        # Taking only those subjects in the included subjects file
        if ext_subjects:
            self.clinic_data = self.clinic_data[self.clinic_data.Guid.isin(self.included_cases.Guid)]

        # Using cleaned version of data (fewer number of subjects)
        if self.cleaned_version:
            dropping_cols = ['Guid','EDComplEventHypoxia','EDComplEventHypotension','EDComplEventAsp',
                             'PresIntubation','EDDrugScreenCocaine','EDDrugScreenAmph','EDDrugScreenPCP',
                             'EDDrugScreenCannabis','EDDrugScreenMethadone','EDDrugScreenBenzo','EDDrugScreenBarb',
                             'EDDrugScreenOpioids','EDArrPupilReactivityLghtLftEyeReslt',
                             'EDArrPupilReactivityLghtRtEyeReslt','GCS']
            self.clinic_data.drop(dropping_cols, axis=1, inplace=True)
            self.clinic_data.rename(columns={'Guid_cleaned': 'Guid', 'GCS_cleaned': 'GCS'}, inplace=True)
            self.clinic_data.dropna(axis=0, subset=['Guid'], inplace=True)

        # Removing missing columns
        columns = self.clinic_data.columns
        for i in columns:
            if (self.clinic_data[i].isnull().sum() > miss_rate * self.clinic_data.shape[0]) \
                    and i != 'GOSEScore':
                self.clinic_data = self.clinic_data.drop(i, 1)
                if self.verbose: print('Column ' + i + ' has been dropped')

        # Removing some columns manually
        dropping_cols = ['EDDrugScreenAlcoholDone', 'GcsEDArrScore', 'GcsCompScoresArrivUnavail', 'Cohort',
                         'EDHematocrit', 'PresSeizures', 'EDDischRespRateType', 'EDArrRespRate', 'EDArrRespRateType', 'EDDischRespRate',
                         'ApptResultOutcomes6Mo', 'PresHypotension', 'PresHypoxia',
                         'LPup', 'LExcl', 'LReact', 'RPup', 'RExcl', 'RReact', 'mGCS_num']
        
        dropping_cols = list(np.intersect1d(dropping_cols, self.clinic_data.columns))
        dropping_cols.extend(self.clinic_data.columns[
                                self.clinic_data.columns.to_series().str.contains('IMPACT')])
        dropping_cols.extend(self.clinic_data.columns[
                                self.clinic_data.columns.to_series().str.contains('SI')])
        dropping_cols.extend(self.clinic_data.columns[
                                self.clinic_data.columns.to_series().str.contains('Mon')])
        if not self.cleaned_version:
            dropping_cols.extend(self.clinic_data.columns[
                                     self.clinic_data.columns.to_series().str.contains('_cleaned')])
        self.clinic_data.drop(dropping_cols, axis=1, inplace=True)
                 
        # Converting hospital discharge time since injury to numeric values
        time = self.clinic_data['HospDischTimeSinceInj'].str.split(' days ', expand=True)
        hour = pd.to_numeric(time[1].str.split(':', expand=True)[0])
        minute = pd.to_numeric(time[1].str.split(':', expand=True)[1])
        self.clinic_data['HospDischTimeSinceInj'] = pd.to_numeric(time[0]) * 24 + hour + minute / 60
        
        # Convert string to float
        self.clinic_data.GOSEScore = self.clinic_data.GOSEScore.apply(float)
        
        #Exclude the outcomes from the clustering process (Include the outcomes when evaluating the TBI phenotypes)
        self.clinic_data.drop(['GOSEScore', 'HospDischTimeSinceInj'], axis=1, inplace=True)
        
        # Indicating float, categorical and binary columns
        cat_cols = self.clinic_data.columns[(self.clinic_data.dtypes == 'object') & (self.clinic_data.columns != 'Guid')]
        flt_cols = self.clinic_data.columns[(self.clinic_data.dtypes == 'float64') | (self.clinic_data.dtypes == 'int64')]
        bin_cols = [i for i in cat_cols if self.clinic_data[i].dropna().unique().shape[0] == 2]
        cat_cols = list(set(cat_cols) - set(bin_cols))
        
        if self.verbose: print('Number of clinic variables before transformation: ' + str(self.clinic_data.shape[1]))
        
        # Handling binary columns
        for i in bin_cols:
            mapping = {self.clinic_data[i].dropna().unique()[0]:0,
                       self.clinic_data[i].dropna().unique()[1]:1}
            self.clinic_data[i] = self.clinic_data[i].map(mapping)
        
        # One-hot vectors for categorical variables
        if len(cat_cols) != 0:
            dummy_df = pd.get_dummies(self.clinic_data[cat_cols[0]], prefix=cat_cols[0])
            dummy_df.loc[self.clinic_data[cat_cols[0]].isnull(), dummy_df.columns.str.startswith(cat_cols[0] + "_")] = np.nan
            for i in cat_cols[1:]:
                dummy_df = pd.concat([dummy_df, pd.get_dummies(self.clinic_data[i], prefix=i)], axis=1)
                dummy_df.loc[self.clinic_data[i].isnull(), dummy_df.columns.str.startswith(i + "_")] = np.nan
            self.clinic_data = pd.concat([self.clinic_data, dummy_df], axis=1)
            self.clinic_data.drop(cat_cols, axis=1, inplace=True)
        
        if self.verbose: print('Number of clinic variables after transformation: ' + str(self.clinic_data.shape[1]))
        
        # Handling outliers
        self.clinic_data.loc[self.clinic_data.AgeRecodedPHI > 120, 'AgeRecodedPHI'] = np.nan

        # Dropping patients with higher gcs than max_gcs
        self.clinic_data = self.clinic_data[(self.clinic_data.GCS <= max_gcs) | (self.clinic_data.GCS.isna())]
        
 
    def clean_vital_data(self):
        
        # Dropping some columns
        dropping_cols = ['DVRr', 'DVArterialPPCO2Val', 'DVCircSuppTyp', 'Cohort', 'DailyVitalsID']
        self.vital_data.drop(dropping_cols, axis=1, inplace=True)

        # Mapping some columns
        mapping = {'Bag mask vent (BVM)': 'NIPPV',
                   'CPAP': 'NIPPV',
                   'BiPAP': 'NIPPV',
                   'No support needed': 'Non-Intubated',
                   'Oral airway': 'Non-Intubated'}
        self.vital_data['DVRespSuppTyp'] = self.vital_data['DVRespSuppTyp'].replace(mapping)

        # Dropping non-numeric rows of labs
        self.vital_data = self.vital_data[self.vital_data.DVHR.apply(lambda x: isfloat(x))]
        # self.vital_data = self.vital_data[self.vital_data.DVRr.apply(lambda x: isfloat(x))]
        self.vital_data = self.vital_data[self.vital_data.DVSpO2.apply(lambda x: isfloat(x))]
        self.vital_data = self.vital_data[self.vital_data.DVSBP.apply(lambda x: isfloat(x))]
        self.vital_data = self.vital_data[self.vital_data.DVDBP.apply(lambda x: isfloat(x))]
        self.vital_data = self.vital_data[self.vital_data.DvTemp.apply(lambda x: isfloat(x))]

        # Dropping rows with unusual time variable
        drop_ind = self.vital_data[~ (self.vital_data['VitalsTimeSinceInj'].str.contains('days', na=False))].index
        self.vital_data.drop(drop_ind, axis=0, inplace=True)
        
        if self.verbose: print('Number of vital variables before transformation: ' + str(self.vital_data.shape[1]))
        
        # One-hot vector for categorical columns
        self.vital_data = pd.concat([self.vital_data, pd.get_dummies(self.vital_data.DVRespSuppTyp, prefix='DVResTyp')], axis=1)
        self.vital_data.drop(['DVRespSuppTyp'], axis=1, inplace=True)
        
        if self.verbose: print('Number of vital variables after transformation: ' + str(self.vital_data.shape[1]))

        # Converting time since injury to numeric values
        time = self.vital_data['VitalsTimeSinceInj'].str.split(' days ', expand=True)
        hour = pd.to_numeric(time[1].str.split(':', expand=True)[0])
        minute = pd.to_numeric(time[1].str.split(':', expand=True)[1])
        self.vital_data['VitalsTimeSinceInj'] = pd.to_numeric(time[0]) * 24 + hour + minute / 60

    def clean_lab_data(self):
        # Dropping some of the columns
        dropping_cols = ['SubjectID', 'DailyLabsID', 'DLNotes', 'DLOther',
                         'DLOtherOtherUnitsSpecify', 'DLDdimers', 'SubjectID.1',
                         'DLFdp']
        dropping_cols.extend(self.lab_data.columns[
                                 self.lab_data.columns.to_series().str.contains('SI')])
        dropping_cols.extend(self.lab_data.columns[
                                 self.lab_data.columns.to_series().str.contains('NotDone')])
        self.lab_data.drop(dropping_cols, axis=1, inplace=True)

        # Dropping rows with unusual time variable
        drop_ind = self.lab_data[~ (self.lab_data['DLTimeSinceInj'].str.contains('days', na=False))].index
        self.lab_data.drop(drop_ind, axis=0, inplace=True)
        
        if self.verbose: print('Number of lab variables before transformation: ' + str(self.lab_data.shape[1]))

        # Converting time since injury to numeric values
        time = self.lab_data['DLTimeSinceInj'].str.split(' days ', expand=True)
        hour = pd.to_numeric(time[1].str.split(':', expand=True)[0])
        minute = pd.to_numeric(time[1].str.split(':', expand=True)[1])
        self.lab_data['DLTimeSinceInj'] = pd.to_numeric(time[0]) * 24 + hour + minute / 60
        
        
    def clean_gcs_data(self):
        # Dropping some of the columns
        dropping_cols = ['DVPupilRtEyeMeasr', 'DVPupilLftEyeMeasr',
                         'DVPupilShapeLftEyeTyp', 'DVPupilShapeRtEyeTyp', 
                         'DVPupilLftEyeMeasrUnkUnt', 'DVPupilRtEyeMeasrUnkUnt',
                         'Cohort', 'Cohort2Wk', 'DVGCSScore', 'DailyVitalsID']
        self.gcs_data.drop(dropping_cols, axis=1, inplace=True)

        # Mapping some values for pupil reactivity
        mapping = {'Sluggish': 'Nonreactive'}
        self.gcs_data['DVPupilReactivityLghtLftEyeReslt'] = self.gcs_data['DVPupilReactivityLghtLftEyeReslt'].replace(mapping)
        self.gcs_data['DVPupilReactivityLghtRtEyeReslt'] = self.gcs_data['DVPupilReactivityLghtRtEyeReslt'].replace(mapping)

        # Combining pupil reactivity columns together
        def aggregate_pupils(row):
            res = np.nan
            if row['DVPupilReactivityLghtLftEyeReslt'] == 'Untestable' or row[
                'DVPupilReactivityLghtRtEyeReslt'] == 'Untestable':
                res = 'Untestable'
            elif row['DVPupilReactivityLghtLftEyeReslt'] == 'Brisk' or row[
                'DVPupilReactivityLghtRtEyeReslt'] == 'Brisk':
                res = 'Both'
            elif row['DVPupilReactivityLghtLftEyeReslt'] == 'Nonreactive' and row[
                'DVPupilReactivityLghtRtEyeReslt'] == 'Nonreactive':
                res = 'Neither'
            elif row['DVPupilReactivityLghtLftEyeReslt'] == 'Nonreactive' or row[
                'DVPupilReactivityLghtRtEyeReslt'] == 'Nonreactive':
                res = 'One'
            return res

        self.gcs_data['PupilReactivity'] = self.gcs_data.apply(aggregate_pupils, axis=1)
        
        if self.verbose: print('Number of GCS variables before transformation: ' + str(self.gcs_data.shape[1]))
        
        # One-hot vector for categorical columns
        dummy_df = pd.get_dummies(self.gcs_data.PupilReactivity, prefix='PupilReactivity')
        dummy_df = pd.concat([dummy_df, pd.get_dummies(self.gcs_data.DVGCSEyes, prefix='GCSEye')], axis=1)
        dummy_df = pd.concat([dummy_df, pd.get_dummies(self.gcs_data.DVGCSMotor, prefix='GCSMtr')], axis=1)
        dummy_df = pd.concat([dummy_df, pd.get_dummies(self.gcs_data.DVGCSVerbal, prefix='GCSVrb')], axis=1)
        self.gcs_data = pd.concat([self.gcs_data, dummy_df], axis=1)
        dropping_cols = ['DVPupilReactivityLghtLftEyeReslt',
                         'DVPupilReactivityLghtRtEyeReslt', 'DVGCSEyes', 'DVGCSMotor', 'DVGCSVerbal', 'PupilReactivity',
                         'DVPupilReactivityLghtLftEyeReslt', 'DVPupilReactivityLghtRtEyeReslt']
        self.gcs_data.drop(dropping_cols, axis=1, inplace=True)
        
        if self.verbose: print('Number of GCS variables after transformation: ' + str(self.gcs_data.shape[1]))
        
        # Dropping rows with unusual time variable
        drop_ind = self.gcs_data[~ (self.gcs_data['DailyGCSTimeSinceInj'].str.contains('days', na=False))].index
        self.gcs_data.drop(drop_ind, axis=0, inplace=True)
        
        # Converting time since injury to numeric values
        time = self.gcs_data['DailyGCSTimeSinceInj'].str.split(' days ', expand=True)
        hour = pd.to_numeric(time[1].str.split(':', expand=True)[0])
        minute = pd.to_numeric(time[1].str.split(':', expand=True)[1])
        self.gcs_data['DailyGCSTimeSinceInj'] = pd.to_numeric(time[0]) * 24 + hour + minute / 60

    def time_series(self, time_stamp=1, only_icu=False):
        # Fining common patients among all
        guid1 = self.lab_data.Guid.unique()
        guid2 = self.vital_data.Guid.unique()
        guid3 = self.gcs_data.Guid.unique()
        guid_all = np.union1d(np.union1d(guid1, guid2),  guid3)

        # Extracting data frame
        df_lab = self.lab_data.copy()
        df_lab = df_lab[df_lab.Guid.isin(guid_all)]
        # ----
        df_vital = self.vital_data.copy()
        df_vital = df_vital[df_vital.Guid.isin(guid_all)]
        # ----
        df_gcs = self.gcs_data.copy()
        df_gcs = df_gcs[df_gcs.Guid.isin(guid_all)]

        # list of variables
        var_lab = list(df_lab.columns)
        var_lab.remove('Guid')
        var_lab.remove('DLTimeSinceInj')
        # ----
        var_vital = list(df_vital.columns)
        var_vital.remove('Guid')
        var_vital.remove('VitalsTimeSinceInj')
        # ----
        var_gcs = list(df_gcs.columns)
        var_gcs.remove('Guid')
        var_gcs.remove('DailyGCSTimeSinceInj')

        # Converting time to time stamps
        df_lab.DLTimeSinceInj = (df_lab.DLTimeSinceInj/time_stamp).apply(np.floor).apply(int)
        # ----
        df_vital.VitalsTimeSinceInj = (df_vital.VitalsTimeSinceInj / time_stamp).apply(np.floor).apply(int)
        # ----
        df_gcs.DailyGCSTimeSinceInj = (df_gcs.DailyGCSTimeSinceInj / time_stamp).apply(np.floor).apply(int)

        # Changing wide data frame to long format
        df_lab = df_lab.melt(id_vars=['Guid', 'DLTimeSinceInj'])
        df_lab.dropna(axis=0, subset=['value'], inplace=True)
        # ----
        df_vital = df_vital.melt(id_vars=['Guid', 'VitalsTimeSinceInj'])
        df_vital = df_vital.rename(columns={'VitalsTimeSinceInj': 'DLTimeSinceInj'})
        df_vital.dropna(axis=0, subset=['value'], inplace=True)
        df_vital.value = df_vital.value.apply(float)
        # ----
        df_gcs = df_gcs.melt(id_vars=['Guid', 'DailyGCSTimeSinceInj'])
        df_gcs = df_gcs.rename(columns={'DailyGCSTimeSinceInj': 'DLTimeSinceInj'})
        df_gcs.dropna(axis=0, subset=['value'], inplace=True)

        # Combining all data frames
        df_ts = pd.concat([df_lab, df_vital, df_gcs], axis=0)

        # aggregating values in each time interval
        df_ts = df_ts.groupby(['Guid', 'DLTimeSinceInj', 'variable']).agg({'value': np.mean})
        df_ts.reset_index(inplace=True)

        # Removing very big time steps
        df_ts = df_ts[df_ts.DLTimeSinceInj < 1000]

        # Filtering data for ICU patients
        if only_icu:
            guid_icu = self.clinic_data.Guid[self.clinic_data['PatientType_Hospital admit with ICU'] == 1]
            guid_hos = self.clinic_data.Guid[self.clinic_data['PatientType_Hospital admit no ICU'] == 1]

            # Building a data frame with value of maximum measurement intervals for each patient
            df_ts_no_icu = df_ts.loc[df_ts.Guid.isin(guid_hos)]
            df_ts_no_icu = df_ts_no_icu[
                (df_ts_no_icu.variable == 'DVSBP') & (df_ts_no_icu.DLTimeSinceInj < 48 / time_stamp)]
            df_ts_no_icu.sort_values(by=['Guid', 'DLTimeSinceInj'], inplace=True, ignore_index=True)

            for i in range(df_ts_no_icu.shape[0]):
                if i == 0:
                    df_ts_no_icu.loc[i, 'time_dif'] = 0
                elif df_ts_no_icu.loc[i, 'Guid'] == df_ts_no_icu.loc[i - 1, 'Guid']:
                    df_ts_no_icu.loc[i, 'time_dif'] = df_ts_no_icu.loc[i, 'DLTimeSinceInj'] - df_ts_no_icu.loc[
                        i - 1, 'DLTimeSinceInj']
                else:
                    df_ts_no_icu.loc[i, 'time_dif'] = 0

            df_ts_no_icu = df_ts_no_icu.groupby(['Guid']).agg({'time_dif': np.max})
            df_ts_no_icu.reset_index(inplace=True)

            # extracting icu patients
            guid_might_icu = df_ts_no_icu[df_ts_no_icu.time_dif<=4].Guid
            guid_all_icu = np.union1d(guid_might_icu, guid_icu)

            df_ts = df_ts[df_ts.Guid.isin(guid_all_icu)]

        # Saving data
        df_ts.to_csv(DATA_PATH + "time_series_data.csv", index=False)

        return df_ts

    def vital_freq(self, hours=48, time_stamp=1):
        df_ts = pd.read_csv(DATA_PATH + "time_series_data.c sv", skipinitialspace=True)

        df_ts_no_icu = df_ts.loc[:]
        df_ts_no_icu = df_ts_no_icu[
            (df_ts_no_icu.variable.isin(self.vital_data.columns)) & (df_ts_no_icu.DLTimeSinceInj < hours / time_stamp)]

        df_ts_no_icu = df_ts_no_icu.groupby(['Guid', 'DLTimeSinceInj']).size().reset_index(name='Freq')

        df_ts_no_icu.sort_values(by=['Guid', 'DLTimeSinceInj'], inplace=True, ignore_index=True)

        df_ts_no_icu = df_ts_no_icu.groupby(['Guid']).count().reset_index()[['Guid', 'Freq']]

        df_ts_no_icu.to_csv(DATA_PATH + "vital_freq.csv", index=False)

        return df_ts_no_icu


