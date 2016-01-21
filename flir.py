from __future__ import print_function

__author__ = 'Nick Steiner'
__version__ = "1r0"
__email__ = "nsteiner@ccny.cuny.edu"

"""
FLIR Library Processing
"""


import os
import sys
import time
import copy
import string
import datetime
import pickle
import logging
from contextlib import contextmanager
from optparse import OptionParser

import numpy as np
from pandas import DataFrame, to_datetime
from netCDF4 import Dataset
from sqlalchemy import types, Column, MetaData, engine, orm, ForeignKey, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import fliratt


'''
----------------------------------
Timezone Fix
----------------------------------
'''

APERTURE = 40.45  # DEGREE
TAN_THT = 2 * np.tan(np.deg2rad(APERTURE / 2.))

IFOV = lambda H: H * 0.72e-3  # [rad/pixel]


'''
----------------------------------
Timezone Fix
----------------------------------
'''

os.environ['TZ'] = 'GMT'
if os.name is 'posix':
    time.tzset()

'''
----------------------------------
SQLAlchemy
----------------------------------

Using SQLLite as database file.
Updated using method: update_database

'''

_PATH = os.path.dirname(os.path.abspath(__file__))
DAT_PATH = os.path.join(_PATH, 'dat')
FLIR_DAT_PATH = os.path.join(DAT_PATH, 'flir')
DADS_DAT_PATH = os.path.join(DAT_PATH, 'dads')
ENGINE = "sqlite:///" + os.path.join(_PATH, 'flir.db')
ee = engine.create_engine(ENGINE)
metadata = MetaData(bind=ee)
Session = sessionmaker(bind=ee)
Base = declarative_base(metadata=metadata)

_kml_dir = os.path.join(_PATH, 'kml') # (? Moved ?)


'''
----------------------------------
Misc. Time Functions
----------------------------------
'''

_dt_frompath = lambda x: datetime.datetime.strptime(os.path.split(x)[-1], '%Y%m%d').date()
_dt_parse = lambda x: datetime.datetime.strptime(x, '%Y%j:%H:%M:%S.%f')
TS2DT = lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')

FLOAT_TIME_OFFSET = np.array(['1980-01-06T00:00:00+0000'], dtype='datetime64[s]').astype('double')
DADS_TIME_OFFSET = np.array(['1980-01-06'], dtype='datetime64[D]')
DF2DADS = lambda index: np.array(index, dtype='datetime64[us]').astype('datetime64[s]').astype('double') - FLOAT_TIME_OFFSET


'''
----------------------------------
Logging
----------------------------------
'''

logging.basicConfig(level=logging.INFO, filename='flir.log',
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


'''
----------------------------------
Database Functions
----------------------------------
'''


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def get_record(record_name):
    with session_scope() as session:
        record = session.query(Record).filter(Record.RecordName == record_name).first()
        session.expunge_all()
    return record


def get_all_objects(session, flir_class):
    """
    Get list of all class instances.

    :param session: Open sqlalchemy session
    :param flir_class: Mapped FLIR Class
    :return: List of FLIR Class instances (e.g. Flights, Records)
    """
    object_list = session.query(flir_class).all()
    session.expunge_all()
    return object_list


def update_database():
    """
    This will update the sqllite file to reflect current data.

    :return:
    """
    pass


def get_flir():
    """


    :return: Flir object.
    """
    with session_scope() as session:
        return Flir(session)



def get_all_flightdates():
    """


    :return: all dates
    """
    with session_scope() as session:
        results = session.query(Flight.StartDate).all()
        session.expunge_all()
    return results


def get_records_byflightid(flight_id):
    with session_scope() as session:
        results = session.query(Record).join(Flight).filter(Flight.FlightID == flight_id). \
            order_by(Record.FlightDate).all()
        session.expunge_all()
    return results

def get_flight_bytime(start_time):
    with session_scope() as session:
        results = session.query(Flight).filter(and_(
                Flight.StartDateTime <= start_time,
                Flight.EndDateTime >= start_time)).first()
        session.expunge_all()
    return results


def get_flight_byflightid(flight_id):
    with session_scope() as session:
        results = session.query(Flight).filter(Flight.FlightID == flight_id).first()
        session.expunge_all()
    return results


def get_flight_bydate(date_obj):
    return get_flight_byflightid(get_flightid_bydate(date_obj))


def get_flightid_bydate(date_obj):
    with session_scope() as session:
        results = session.query(Flight.FlightID).filter(Flight.StartDate == date_obj).first()
        session.expunge_all()
    if results:
        results = results[0]
    else:
        print(date_obj)
    return results

def get_flightid():
    """


    :return: flight id
    """
    with session_scope() as session:
        results = session.query(Flight.FlightID).all()
        session.expunge_all()
    return results


def read_dads_data(dads_file, variable_definitions):
    with Dataset(dads_file, 'r') as ncdf_obj:
        array_list = []
        for variable, (group, dtype) in variable_definitions.iteritems():
            array_list.append(np.array(ncdf_obj.groups[group].variables[variable][:], dtype=dtype))
    return np.core.records.fromarrays(array_list, names=variable_definitions.keys())


def dump_flir_dates():
    with session_scope() as session:
        flir_dates = session.query(Flight.StartDate).all()
    flir_dates = [flir_date[0] for flir_date in flir_dates]
    flir_list = zip(range(1, len(flir_dates) + 1), flir_dates)
    print(flir_list)
    with open(os.path.join(_PATH, 'flir_dates.pck'), 'w') as file:
        pickle.dump(flir_list, file)


'''
----------------------------------
FLIR Data Classes
----------------------------------
'''


class Flir(object):
    def __init__(self, session=None):

        """
        Should initialize with OCR objects from file. Flights --> Records

        :type session: orm session, if None Flir only finds local files
        """
        # set file lists
        self.record_file_list = 'sfmov'
        self.dad_file_list = 'nc'
        if session:
            self.flight_list = session
            self.record_list = session

    @property
    def record_file_list(self):
        """
        List of the current record files

        :return: list of sfmov files with full path
        """
        return self._record_file_list

    @record_file_list.setter
    def record_file_list(self, filter_string):
        self._record_file_list = self._get_file_list(filter_string, FLIR_DAT_PATH)

    @property
    def dad_file_list(self):
        """
        List of the current Flight files.

        :return: list of dads files with full path
        """
        return self._dad_file_list

    @dad_file_list.setter
    def dad_file_list(self, filter_string):
        self._dad_file_list = self._get_file_list(filter_string, DADS_DAT_PATH)

    @staticmethod
    def _get_file_list(file_filter, dat_path=DAT_PATH):
        """
        Get files in path using the filename filter

        :param file_filter: String file ending used for selecting file.
        :return: List of files with full path.
        """
        file_list = []
        for (root, dirs, files,) in os.walk(dat_path, followlinks=True):
            for file_ in files:
                if file_.endswith('.{}'.format(file_filter)):
                    file_list.append(os.path.join(root, file_))
        return file_list

    @property
    def flight_list(self):
        return self._flight_list

    @flight_list.setter
    def flight_list(self, session):
        self._flight_list = get_all_objects(session, Flight)

    @property
    def record_list(self):
        return self._record_list

    @record_list.setter
    def record_list(self, session):
        self._record_list = get_all_objects(session, Record)


    def file_dict(self, record_name):
        """
        :param record_name:
        :return:
        """
        return dict([(ext, '{}.{}'.format(record_name, ext)) for ext in ['sfmov', 'pod', 'inc']])

    def _get_path(self, date_obj):
        return os.path.join(DAT_PATH, date_obj.strftime('%Y%m%d'))

    def _dt_parse(self, year, date_str):
        return datetime.datetime.strptime(str(year) + date_str, '%Y%j:%H:%M:%S.%f')


LOAD_ON_INIT = False

class Flight(Base):
    __tablename__ = 'flight'

    FlightID = Column(types.String, primary_key=True)
    Version = Column(types.String)
    CollectionLabel = Column(types.String)
    LongName = Column(types.String)
    ShortName = Column(types.String)
    StartDateTime = Column(types.DateTime)
    EndDateTime = Column(types.DateTime)
    StartDate = Column(types.Date)
    EndDate = Column(types.Date)

    _time_offset = 315964800

    _dads_variables = {
        'center_lat':                ('geolocation',         'single'),
        'center_lat_standard_error': ('geolocation',         'single'),
        'center_lon':                ('geolocation',         'single'),
        'center_lon_standard_error': ('geolocation',         'single'),
        'height':                    ('geolocation',         'single'),
        'height_standard_error':     ('geolocation',         'single'),
        'geolocation_qc':            ('geolocation',         'uint8'),
        'time':                      ('geolocation',         'double'),
        'pitch':                     ('science_measurement', 'single'),
        'pitch_qc':                  ('science_measurement', 'uint8'),
        'roll':                      ('science_measurement', 'single'),
        'roll_qc':                   ('science_measurement', 'uint8'),
        'heading':                   ('science_measurement', 'single'),
        'heading_qc':                ('science_measurement', 'uint8')}

    _dads_attributes = {
        'Version':         lambda x: getattr(x, 'build_id'),
        'CollectionLabel': lambda x: getattr(x, 'collection_label'),
        'LongName':        lambda x: getattr(x, 'long_name'),
        'ShortName':       lambda x: getattr(x, 'long_name'),
        'StartDateTime':   lambda x: TS2DT(getattr(x, 'data_start_time')),
        'EndDateTime':     lambda x: TS2DT(getattr(x, 'data_stop_time')),
        'StartDate':       lambda x: TS2DT(getattr(x, 'data_start_time')).date(),
        'EndDate':         lambda x: TS2DT(getattr(x, 'data_stop_time')).date()}

    def __init__(self, dads_file, load_on_init=LOAD_ON_INIT):
        for att, value in Flight.read_dads_attributes(dads_file).iteritems():
            setattr(self, att, value)
        if load_on_init:
            self.load()

    @orm.reconstructor
    def init_on_load(self):
        if LOAD_ON_INIT:
            self.load()

    def load(self):
        self.data_array = self.file_path
        self.set_geolocation()
        #self.record_list = get_records_bytime(self.StartDateTime, self.EndDateTime)
        self.record_list = get_records_byflightid(self.FlightID)

    @property
    def file_path(self):
        return os.path.join(_PATH, 'dat', 'dads', self.FlightID + '.nc')

    @staticmethod
    def get_record_names(flight_date_str, flir_dat_path=FLIR_DAT_PATH):
        record_list = []
        for (root, dirs, files,) in os.walk(os.path.join(flir_dat_path, flight_date_str)):
            for file_ in files:
                if file_.endswith('.' + 'sfmov'):
                    record_list.append(Record.parse_record_file(file_)['Name'])
        return record_list

    @staticmethod
    def read_dads_attributes(dads_file):
        dads_metadata = {}
        Flight.test_file(dads_file)
        with Dataset(dads_file, 'r') as ncdf_obj:
            for att, metadata_function in Flight._dads_attributes.iteritems():
                dads_metadata[att] = metadata_function(ncdf_obj)
        path, file_ = os.path.split(dads_file)
        dads_metadata.update({'FlightID': file_.rstrip('.nc')})
        return dads_metadata

    def __repr__(self):
        return '<FLIR-{};{}>'.format(self.FlightID, self.StartDate)

    @staticmethod
    def test_file(dads_file):
        try:
            test_ = Dataset(dads_file, 'r')
            test_.close()
        except Exception as e:
            logging.error('Corrupt netcdf file: {}'.format(dads_file))
            raise Exception('Corrupt netcdf file: {}'.format(dads_file))


    @property
    def data_array(self):
        return self._data_array

    @data_array.setter
    def data_array(self, dads_file):
        self._data_array = read_dads_data(dads_file, self._dads_variables)

    @property
    def geolocation(self):
        return self._geolocation

    def set_geolocation(self):
        record_array = self.data_array
        geolocation_frame = DataFrame.from_records(record_array)
        us_array, s_array = np.modf(self.data_array['time'])
        index_ = DADS_TIME_OFFSET.astype('datetime64[us]') + \
                     s_array.astype('timedelta64[s]').astype('timedelta64[us]') + \
                     (us_array * 1e6).astype('timedelta64[us]')
        geolocation_frame.index = index_
        self._geolocation = geolocation_frame

    def parse_dads_time_str(self, time_str):
        return datetime.datetime.strptime(str(self.StartDate.year) + time_str, '%Y%j:%H:%M:%S.%f')

    def get_record_time(self, record_obj):
        '''

        Merges the flight navigation system with the frame using interpolation.
        '''
        geolocation = self.geolocation.copy()
        record_time_series = np.array([self.parse_dads_time_str(time_) for time_ in record_obj.reduce_frame['Time']], dtype='datetime64[us]')
        for time_ in record_time_series:
            geolocation.loc[time_.astype(datetime.datetime)] = [np.nan for i in geolocation.columns]
        return geolocation.interpolate(method='time').loc[record_time_series]


class Record(Base):
    name = 'record'
    _BAD_PIX = [( )]
    __tablename__ = 'record'

    RecordName = Column(types.String, primary_key=True)
    RecordFile = Column(types.String)
    FlightID = Column(types.String, ForeignKey("flight.FlightID"))
    FlightDate = Column(types.DateTime)
    # from sfmov file
    AdBits = Column(types.String)
    XPixels = Column(types.Integer)
    YPixels = Column(types.Integer)
    DataType = Column(types.String)
    HdSize = Column(types.String)
    KeyWord = Column(types.String)
    BPFile = Column(types.String)
    IncludeFile = Column(types.String)
    ReduceFile = Column(types.String)
    SubFrame = Column(types.Integer)
    Frames = Column(types.BigInteger)
    # from pod file
    AtmA1 = Column(types.Float)
    AtmA2 = Column(types.Float)
    AtmB1 = Column(types.Float)
    AtmA2 = Column(types.Float)
    AtmX = Column(types.Float)
    B = Column(types.Float)
    BandpassHigh = Column(types.Float)
    BandpassLow = Column(types.Float)
    BGValue = Column(types.Float)
    C0 = Column(types.Float)
    C1 = Column(types.Float)
    C2 = Column(types.Float)
    C3 = Column(types.Float)
    C4 = Column(types.Float)
    C5 = Column(types.Float)
    C6 = Column(types.Float)
    F = Column(types.Float)
    MaxCounts = Column(types.BigInteger)
    MaxRadiance = Column(types.Float)
    MaxTemperature = Column(types.Float)
    MinCounts = Column(types.BigInteger)
    MinRadiance = Column(types.Float)
    MinTemperature = Column(types.Float)
    PolynomialOrder = Column(types.Integer)
    R = Column(types.Float)
    TempC0 = Column(types.Float)
    TempC1 = Column(types.Float)
    TempC2 = Column(types.Float)
    TempC3 = Column(types.Float)
    TempC4 = Column(types.Float)
    TempC5 = Column(types.Float)
    TempC6 = Column(types.Float)
    ReduceFrames = Column(types.String)
    PnSize = Column(types.Integer)
    PuSize = Column(types.Integer)
    PcSize = Column(types.Integer)
    NParameters = Column(types.Integer)

    _sfmov_header_size = 2336

    _pRnm = ('nfcC6_0', 'nfcC5_0', 'nfcC4_0', 'nfcC3_0', 'nfcC2_0', 'nfcC1_0', 'nfcC0_0')
    _pTnm = ('nfcTempC6_0', 'nfcTempC5_0', 'nfcTempC4_0', 'nfcTempC3_0', 'nfcTempC2_0', 'nfcTempC1_0', 'nfcTempC0_0')

    _movie_path = 'movie'
    _reference_files = ('Include', 'REFile')

    def __init__(self, record_file, load=LOAD_ON_INIT):

        """
        Inits using record file. CAL, POD, files must be in same directory
        and have same name.
        :param record_file: Full path to record file (SFMOV!!)
        """
        self.RecordName = os.path.basename(record_file).rstrip('.sfmov')
        self.RecordFile = os.path.relpath(record_file)


        try:
            date_obj = datetime.datetime.strptime(os.path.split(os.path.dirname(record_file))[-1], '%Y%m%d').date()
            self.FlightDate = date_obj
            self.FlightID = get_flightid_bydate(date_obj) # first guess
        except:
            pass
        # set metadata
        record_metadata = Record.parse_record_file(record_file, SfmovMetadata())
        self.set_record_metadata(record_metadata)
        self.set_metadata()
        self.set_index()
        try:
            assert self.index is not None
            self.FlightDate = self.index[0].date()
        except:
            logging.error('Metadata ({}) not found or corrupt: {}'. \
                          format(self.ReduceFile, self.RecordName))
        # find record-object by time
        try:
            flight = get_flight_bytime(self.index[0])
            assert flight is not None
            self.FlightID = flight.FlightID
        except :
            logging.error('Cannot find DADS file for {}, date: {}'. \
                          format(self.FlightDate, self.RecordName))
        if load:
            self.load_array()

    @orm.reconstructor
    def init_on_load(self):
        self.set_metadata()
        self.set_index()
        print ('Metadata parsed: {}'.format(self))
        if LOAD_ON_INIT:
            self.load_array()

    def load_array(self):
        try:
            array_ = self._read_array()
        except:
            array_ = self.read_sfmov(self.sfmov_file_path)
        self._count_array = array_

    @property
    def index(self):
        return self._index

    def set_index(self):
        try:
            assert hasattr(self, 'reduce_frame')
            str2dt = lambda t: to_datetime(str(self.FlightDate.year) + t, format='%Y%j:%H:%M:%S.%f')
            self._index = [str2dt(t) for t in self.reduce_frame.Time]
        except Exception as e:
            logging.error('Index not set: {}'.format(self.RecordName))
            print('reduce frame not found !!')
            self._index = None

    @property
    def sfmov_file_path(self):
        return self._gen_path('sfmov')

    @property
    def pod_file_path(self):
        return self._gen_path('pod')

    @property
    def inc_file_path(self):
        return self._gen_path('inc')

    @property
    def sbp_file_path(self):
        return self._gen_path('sbp')

    @property
    def cal_file_path(self):
        return self._gen_path('cal')

    @property
    def source_files(self):
        record_files = []
        for type_ in ['sfmov', 'pod', 'inc', 'sbp', 'cal']:
            record_file = self._gen_path(type_)
            if os.path.exists(record_file):
                record_files.append(os.path.basename(record_file))
        record_files.append(self.FlightID + '.nc')
        return ','.join(record_files)

    def _gen_path(self, type_):
        file_path = os.path.join(_PATH, os.path.dirname(self.RecordFile))
        return os.path.join(file_path, '{}.{}'.format(self.RecordName, type_))

    def set_metadata(self):
        record_metadata = Record.parse_record_file(self.sfmov_file_path, SfmovMetadata())
        if hasattr(record_metadata, 'Include'):
            if os.path.exists(self.inc_file_path):
                include_metadata = Record.parse_record_file(self.inc_file_path, IncludeMetadata())
                self.set_record_metadata(include_metadata)
            else:
                logging.error('Missing Include file: {}'.format(self.inc_file_path))
        else:
            logging.error('Corrupt SFMOV Header: {}'.format(self.sfmov_file_path))

        if hasattr(record_metadata, 'REFile'):
            if os.path.exists(self.pod_file_path):
                reduce_metadata = Record.parse_record_file(self.pod_file_path, ReduceMetadata())
                self.set_record_metadata(reduce_metadata)
                self.set_reduce_frame()
            else:
                logging.error('Missing Reference file: {}'.format(self.pod_file_path))
        else:
            logging.error('Corrupt SFMOV Header: {}'.format(self.sfmov_file_path))

    @staticmethod
    def parse_header_line(line):
        line_str_list = line.split(' ')
        if line.startswith('DATA'):
            data_stop = True
            return None, None, data_stop
        assert len(line_str_list) > 1
        header_key, header_value_str = [line_str_list[0], ' '.join(line_str_list[1:]).rstrip()]
        data_stop = (header_key == 'saf_padding') | (header_key == 'DATA')
        return header_key, header_value_str, data_stop

    @staticmethod
    def parse_record_file(record_file, metadata_obj):
        with open(record_file, 'r') as f:
            while True:
                line = f.readline()
                if line:

                    header_key, header_value_str, data_stop = Record.parse_header_line(line)
                    ''' debugging
                    # print(line)  #debugging
                    if (header_key != 'Group') & (header_value_str is not None):
                        if header_value_str.startswith('Rec-'):
                            print(header_key)
                        metadata_obj[header_key] = header_value_str
                    '''
                    metadata_obj[header_key] = header_value_str
                else:
                    break
                if data_stop:
                    break
        try:
            del metadata_obj['Group']  # not valid metadata
        except:
            pass
        return metadata_obj

    def set_record_metadata(self, metadata_obj):
        for key, (_, attribute_name) in metadata_obj.type_definitions.iteritems():
            if attribute_name:
                try:
                    setattr(self, attribute_name, metadata_obj[key])
                except:
                    setattr(self, attribute_name, None)
                    #print('No key in file: {}'.format(key))

    def __repr__(self):
        return '<FLIR-{};{}>'.format(self.name, self.RecordName)

    @property
    def reduce_frame(self):
        return self._reduce_frame

    def set_reduce_frame(self):
        self._reduce_frame = self.parse_reduce_file(self.pod_file_path)

    @staticmethod
    def parse_reduce_file(reduce_file):
        """
        Read reduce file and return dataframe.
        :param reduce_file: Full path to reduce file.
        :return: pd.DataFrame or Nonetype if file is empty
        """
        with open(reduce_file, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                if file.readline().lower().startswith('data'):
                    break
            file_str = file.read()
        if file_str:
            file_lines = file_str.split('\r\n')
            type_fns = ReduceTypes.types
            record_list = []
            for record_str in file_lines[3:]:
                if record_str:
                    record_list.append([fun(rec) for fun, rec in zip(type_fns, record_str.split('\t'))])
            return DataFrame.from_records(record_list, columns=ReduceTypes.names, index='DptNum')
            #return reduce_frame

    @property
    def data_frame(self):
        if not hasattr(self, '_data_frame'):
            data_frame = self.dads_frame.copy()
            data_frame['counts_mean'] = self.count_array.mean(axis=(1, 2))
            data_frame['counts_std'] = self.count_array.mean(axis=(1, 2))
            data_frame['radiance_mean'] = self.cast_temp(data_frame['counts_mean'])
            data_frame['radiance_std'] = self.cast_temp(data_frame['counts_std'])
            self._data_frame = data_frame
        return self._data_frame

    @property
    def count_array(self):
        return self._count_array

    @property
    def temp_array(self):
        return self.cast_temp(self.count_array)

    @property
    def rad_array(self):
        return self.cast_rad(self.count_array)

    @property
    def rad_coefficients(self):
        return [C for C in [self.C6, self.C5, self.C4, self.C3, self.C2, self.C1, self.C0] if C]

    @property
    def temp_coefficients(self):
        return [C for C in [self.TempC6, self.TempC5, self.TempC4, self.TempC3, self.TempC2, self.TempC1, self.TempC0] if C]

    def cast_rad(self, count_array):
        if any(self.rad_coefficients):
            return np.polyval(self.rad_coefficients, count_array)

    def cast_temp(self, count_array):
        if any(self.temp_coefficients):
            return np.polyval(self.temp_coefficients, self.cast_rad(count_array))

    def read_sfmov(self, sfmov_file_path):
        with open(sfmov_file_path, 'rb') as f:
            while True:
                line = f.readline().rstrip()
                if line == 'DATA':
                    break
                if not line:
                    raise ('DATA NOT FOUND BEFORE FILE END!!!')
            array = np.fromfile(f, dtype=np.dtype(self.DataType.lower()))
        array_shape = (self.Frames, self.XPixels, self.YPixels)
        try:
            # case  -->  frames same as in header
            array.resize(array_shape)
        except:
            try:
                frames_calculated = (array.size/(self.XPixels * self.YPixels)) #  chopped if incomplete frames
                #case  -->  all whole frames
                assert array.size % (self.XPixels * self.YPixels) < 1
                array.resize(frames_calculated, self.XPixels, self.YPixels)

            except:
                #case -->  incomplete frames last one
                try:
                    whole_frames = slice(0, -1 * (array.size % (array_shape[1] * array_shape[2])))
                    array = array[whole_frames].reshape(frames_calculated, self.XPixels, self.YPixels)

                except:
                    raise Exception('Cannot Read SFMOV File (CORRUPT?)')
            finally:
                self.Frames = frames_calculated
        return array

    @property
    def geolocation(self):
        return self._geolocation

    def set_geolocation(self, flight):
        try:
            assert hasattr(flight, '_geolocation')
        except Exception as e:
            print('Flight Geolocation not set.')
            raise(e)
        flight_geo = flight.geolocation.copy()
        # Check if record is within flight geolocation
        assert flight_geo.index[0] < self.index[0]
        assert flight_geo.index[-1] > self.index[-1]
        for i in self.index:
            flight_geo.loc[i] = np.nan
        flight_geo.sort_index(inplace=True)
        flight_geo.interpolate(method='time', inplace=True)
        '''  debugging
        start_real = np.where(flight_geo.index == self.index[0])[0][0] - 1
        end_real = np.where(flight_geo.index == self.index[-1])[0][0] + 1
        ax = plt.subplot(111)
        flight.geolocation['height'][flight_geo.index[start_real]:flight_geo.index[end_real]].plot(ax=ax, style='ok')
        flight_geo['height'].loc[self.index[0]:self.index[-1]].plot(style='.r', ax=ax)
        '''
        self._geolocation = flight_geo.loc[self.index]


'''
----------------------------------
FLIR Record File Metadata
----------------------------------
'''

class FileMetadata(object):
    type_definitions = {}

    def __call__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if key in self.type_definitions:
            set_type, _ = self.type_definitions[key]
            try:
                typed_value = set_type(value)
            except:
                typed_value = None
            self.__dict__[key] = typed_value

    def __getitem__(self, key):
        return self.__dict__[key]


class SfmovMetadata(FileMetadata):
    type_definitions = {
        'HdSize':   (str, None),  #
        'ADBits':   (str, 'AdBits'),  # AdBits
        'xPixls':   (int, 'XPixels'),
        'yPixls':   (int, 'YPixels'),
        'DaType':   (str, 'DataType'),
        'KeyWrd':   (str, 'KeyWord'),
        'BPFile':   (str, 'BPFile'),
        'Include':  (str, 'IncludeFile'),
        'REFile':   (str, 'ReduceFile'),
        'SUBFRAME': (int, 'SubFrame'),  # ??
        'NumDPs':   (int, 'Frames')}


class IncludeMetadata(FileMetadata):
    type_definitions = {
        "HdSize":            (str,   None),
        "FRate_0":           (int,   'FrameRate'),  #   30 Frame FrameRate
        "ITime_0":           (float, 'IntegrationTime'),  #  6.68608 IntegrationTime
        "nfcAtmA1_0":        (float, 'AtmA1'),  #  4.82300000000000e-003 AtmA1
        "nfcAtmA2_0":        (float, 'AtmA2'),  #  1.36874000000000e-001 AtmA2
        "nfcAtmB1_0":        (float, 'AtmB1'),  #  2.95600000000000e-003  AtmB1
        "nfcAtmB2_0":        (float, 'AtmA2'),  #  1.22520000000000e-002 AtmA2
        "nfcAtmX_0":         (float, 'AtmX'),  #  9.00408000000000e-001 AtmX
        "nfcB_0":            (float, 'B'),  #  3.13772860649615e+003 B
        "nfcBandpassHigh_0": (float, 'BandpassHigh'),  #  4.90000000000000e+000 BandpassHigh
        "nfcBandpassLow_0":  (float, 'BandpassLow'),  #  3.00000000000000e+000 BandpassLow
        "nfcBGValue_0":      (float, 'BGValue'),  #  0.00000000000000e+000 BGValue
        "nfcC0_0":           (float, 'C0'),  #  -7.45567590398151e-005 C0
        "nfcC1_0":           (float, 'C1'),  #   3.71618922472575e-008 C1
        "nfcC2_0":           (float, 'C2'),  #  -1.04927595246710e-013 C2
        "nfcC3_0":           (float, 'C3'),  #  0.00000000000000e+000 C3
        "nfcC4_0":           (float, 'C4'),  #  0.00000000000000e+000 C4
        "nfcC5_0":           (float, 'C5'),  #  0.00000000000000e+000 C5
        "nfcC6_0":           (float, 'C6'),  #  0.00000000000000e+000 C6
        "nfcF_0":            (float, 'F'),  #  1.00000000000000e+000 F
        "nfcMaxCounts_0":    (int,   'MaxCounts'),  #   13319 MaxCounts
        "nfcMaxRad_0":       (float, 'MaxRadiance'),  #  4.02841513277963e-004  MaxRadiance
        "nfcMaxTemp_0":      (float, 'MaxTemperature'),  #  5.50000000000000e+001 MaxTemperature
        "nfcMinCounts_0":    (int,   'MinCounts'),  #   2634 MinCounts
        "nfcMinRad_0":       (float, 'MinRadiance'),  #  2.37031799770193e-005 MinRadiance
        "nfcMinTemp_0":      (float, 'MinTemperature'),  #  -2.00000000000000e+001 MinTemperature
        "nfcPolyOrder_0":    (int,   'PolynomialOrder'),  #   2 PolynomialOrder
        "nfcR_0":            (float, 'R'),  #  5.72495803288052e+000 R
        "nfcTempC0_0":       (float, 'TempC0'),  #  -4.47282235102836e+001 TempC0
        "nfcTempC1_0":       (float, 'TempC1'),  #  1.23554965553619e+006 TempC1
        "nfcTempC2_0":       (float, 'TempC2')}  #  -1. TempC2


class ReduceMetadata(FileMetadata):
    type_definitions = {
        "NumDPs": (lambda x: -9999 if x == 'auto' else int(x), 'ReduceFrames'),
        "PNSize": (int, 'PnSize'),
        "PUSize": (int, 'PuSize'),
        "PCSize": (int, 'PcSize'),
        'NParam': (int, 'NParameters')}


CHAR_BOOL = lambda x: True if (x == 'T') else False


class ReduceTypes(object):
    type_definitions = (
        ('DptNum',              int),
        ('Time',                str),  #    093:19:21:19.885200
        ('Talo',                float),  #  04.966691
        ('DeltaT',              float),   # 01.000006
        ('FrameNumber',         int),  #    197304
        ('Preset',              int),  #    0
        ('ActivePreset',        int),  #    0
        ('FrameCounter',        int),  #    197304
        ('HeaderID',            int),  #    20
        ('IRIGLocked',          CHAR_BOOL),  #      T
        ('Trigger',             CHAR_BOOL),  #         F
        ('VideoNUCOn',          CHAR_BOOL),  #      T
        ('GigENucOn',           CHAR_BOOL),  #       T
        ('CameraLinkNUCOn',     CHAR_BOOL),  # T
        ('FlagInFOV',           CHAR_BOOL),  #       F
        ('FrontPanelTemp',      float),  #      10.85
        ('AirGapTemp',          float),  #          13.74
        ('InternalTemp',        float),  #        13.52
        ('FPATemp',             float),  #             74.41
        ('IntegrationTime',     float),  #     6.686080
        ('FilterID',            float),  #            0
        ('FPAControlWord',      str),  #        0x6000E4706D53E7F30001
        ('SaturationThreshold', int),  #   15000
        ('SaturatedPixels',     int),  #       18
        ('SaturationFlag',      CHAR_BOOL),  #  F
        ('FrameRate',           float),  #           30.00
        ('DigitalGain',         float),  #         0.996
        ('DigitalOffset',       float),  #       211.0
        ('DetectorType',        int),  #          0
        ('AnalogOffset',        float),  #          0.000
        ('MainBoardTemp',       float),  #          24.93
        ('PowerBoardTemp',      float),  #          22.37
        ('DigitizerBoardTemp',  float),  #          24.12
        ('NUCFlagTemp',         float),  #          12.01
        ('AnalogIn_1',          float),  #          0.007
        ('AnalogIn_2',          float),  #          0.002
        ('AnalogIn_3',          float),  #          0.006
        ('AnalogIn_4',          float),  #          0.000
        ('AnalogIn_5',          float),  #          0.000
        ('AnalogIn_6',          float),  #          0.000
        ('AnalogIn_7',          float),  #          0.000
        ('AnalogIn_8',          float),  #          0.000
        ('DigitalIn',           str),  #             0x55
        ('DigitalOut',          str),  #            0xAA
        ('ADResolution',        int),  #          14
        ('ROICType',            int),  #              5
        ('CameraType',          int),  #            4
        ('CameraSubType',       int),  #         5
        ('CameraSN',            int),  #              25
        ('LensID',              int),  #                0
    )
    types = [type for name, type in type_definitions]
    names = [name for name, type in type_definitions]

'''
----------------------------------
NetCDF Writing
----------------------------------
'''


class _GeneralNetCDFFormat():
    contact = 'Kyle McDonald'
    contact_email = 'kmcdonald2@ccny.cuny.edu'
    institution = 'The City College of the City University of New York',
    source = 'in situ airborne observation'
    references = 'See associated README file.'
    comment = 'See associated README file.'
    Conventions = 'CF-1.6'
    instrument = 'FLIR SC8200; 25 mm lens [#23898-000]'
    general_att_names = ('institution', 'source', 'history', 'references', 'comment', 'Conventions', 'instrument')
    file_ext = 'nc'

    @property
    def name(self):
        dict_ = dict([(f[1], getattr(self, f[1])) for f in string.Formatter().parse(self.name_fmt)])
        return self.name_fmt.format(**dict_)

    @property
    def file_name(self):
        return '{}.{}'.format(self.name, self.file_ext)

    @property
    def folder(self):
        return os.path.join(_PATH, 'dat', self.short_name, self.data_bounds[0].strftime('%Y%m%d'))

    @property
    def file_path(self):
        return os.path.join(self.folder, self.file_name)

    @property
    def name(self):
        dict_ = dict([(f[1], getattr(self, f[1])) for f in string.Formatter().parse(self.name_fmt)])
        return self.name_fmt.format(**dict_)

    @property
    def file_name(self):
        return '{}.{}'.format(self.name, self.file_ext)

    @property
    def history(self):
        args_ = ' '.join(sys.argv)
        time_ = datetime.datetime.utcnow().isoformat()[:-3] + 'Z'
        return '{}_{}'.format(time_, args_)


    def write(self):
        self.make_paths()
        with Dataset(self.file_path, 'w') as ncfile:
        # must have a title or will raise error
            try:
                setattr(ncfile, 'title', getattr(self, 'title'))
            except Exception as e:
                raise(Exception('FLIR products must have a title!'))
            # write general attributes
            for key in self.general_att_names:
                setattr(ncfile, key, getattr(self, key))
            # write product attributes
            self.write_product_attributes(ncfile)
            # write variable dimensions
            dim_vars = self.write_dimensions(ncfile)
            # write geolocation information
            geo_group = self.write_geolocation(ncfile, dim_vars)
            # create radiance group group
            print('loading data array ...')
            self.record.load_array()
            sci_group = self.write_science_measurement(ncfile)
        return sci_group, geo_group, dim_vars



    def make_paths(self):
        try:
            os.makedirs(self.folder)
        except:
            pass


class FLIR01A(_GeneralNetCDFFormat):
    long_name = 'CARVE_FLIR01A'
    short_name = 'FLIR_L1A'
    processing_level = 'Level 1A'
    master_quality_flag = 'Good'
    build_id = 'b23'
    sampling_interval = "Grab"
    frequency_of_sampling = "1 second - < 1 minute"
    specification_name = "Carve Data Bible"
    specification_version = "20150401"
    collection_label = 'Production_B2.3_r01'
    description = 'Earth referenced radiance counts at the Sensor measured by the FLIR imaging camera'
    title = 'CARVE FLIR Level 1A Radiance Counts at the Sensor'
    #name_fmt = 'carve_{short_name}_{build_id}_{flight_date}_{record_number:06g}_{production_date_time}'

    name_fmt = 'carve_{short_name}_{build_id}_{flight_date}_{record_number}_{production_date_time_fname}'
    algorithm_version = '1r0'

    product_att_names = ('product_source','ancillary_file_source', 'collection_label',
                         'data_start_time', 'data_stop_time', 'sampling_interval', 'frequency_of_sampling',
                         'production_date_time','specification_name','specification_version','build_id',
                         'long_name', 'short_name', 'processing_level', 'master_quality_flag', 'algorithm_version',
                         'description')

    def __init__(self, record):
        self.record = record
        self.record_number = record.RecordName.lstrip('Rec-') #self.sorted([r.RecordName for r in get_records_byflightid(record.FlightID)]).index(record.RecordName)
        self.product_source = record.source_files
        self.ancillary_file_source = r'NA'
        self.data_bounds = (record.index[0].to_datetime(), record.index[-1].to_datetime())
        self.data_start_time = self.data_bounds[0].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        self.data_stop_time = self.data_bounds[1].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        self.production_date_time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        self.production_date_time_fname = ''.join(c for c in self.production_date_time if c not in '-:.ZT')
        self.flight_date = self.data_bounds[0].strftime('%Y%m%d')

    def write_geolocation(self, ncfile, dim_vars):
        # create geolocation group group
        geo_group = ncfile.createGroup('geolocation')
        geo_data = self.record.geolocation.copy()
        geo_data['time_stamp'] = [g.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' for g in self.record.index]
        times, rasters, scans = dim_vars
        # times match geolocation variables ???
        times[:] = DF2DADS(geo_data.index)

        #TODO: add units to the dimension variables
        times.units = 'seconds since 1980-01-06T00:00:00.000000'

        for var_name in geo_data.columns:
            var_info = fliratt.geolocation_variables[var_name]
            att_ = fliratt.geolocation_attributes[var_name]
            #print(var_info['fill_value']) #  debug
            nc_var = geo_group.createVariable(var_name, var_info['data_type'], (var_info['dimension'], ), zlib=True,  fill_value=var_info['fill_value'])
            if '_error' not in var_name:
                # set the data
                nc_var[:] = geo_data[var_name].values
                # set the variables
                for key, att_values in att_.iteritems():
                    setattr(nc_var, key, att_values)
        return geo_group

    def write_dimensions(self, ncfile):
        # create dimensions
        time_dim = ncfile.createDimension('time', len(self.record.index))
        raster_dim = ncfile.createDimension('raster', 1024)
        scan_dim = ncfile.createDimension('scan', 1024)
        # create dimension variables
        times = ncfile.createVariable("time", "int64", ("time",), fill_value=255)
        rasters = ncfile.createVariable('raster', 'int16', ('raster', ), fill_value=255)
        scans = ncfile.createVariable('scan', 'int16', ('scan', ), fill_value=255)
        # set dimension variables
        rasters[:] = np.arange(1, 1024 + 1)
        scans[:] = np.arange(1, 1024 + 1)
        rasters.units = 'raster position [n]'
        scans.units = 'scan position [n]'
        return (times, rasters, scans)

    def write_product_attributes(self, ncfile):
        for key in self.product_att_names:
            setattr(ncfile, key, getattr(self, key))

    def write_science_measurement(self, ncfile):
        # create science measurement group
        sc_group = ncfile.createGroup('science_measurement')
        att_dict = fliratt.get_flir01a_attr(self.record)
        var_dict = fliratt.flir01a_sm_variables
        sc_data = {'radiance': self.record.count_array}
        for var_name in var_dict:
            var_info = var_dict[var_name]
            att_ = att_dict[var_name]
            nc_var = sc_group.createVariable(var_name, var_info['data_type'], var_info['dimension'],
                                             zlib=True, fill_value=var_info['fill_value'])
            # set the data
            nc_var[:] = sc_data[var_name]
            # set the variables
            for key, att_values in att_.iteritems():
                if att_values is None:
                    att_values = ''
                setattr(nc_var, key, att_values)
        return sc_group


def write_FLIR01A(overwrite=False, year=None, month=None, day=None):
    for (date_obj,) in get_all_flightdates():
        if year:
            if date_obj.year != year:
                continue
            else:
                if month:
                    if date_obj.month != month:
                        continue
                    else:
                        if day:
                            if date_obj.day !=day:
                                continue
        flight = get_flight_bydate(date_obj)
        flight.load()
        for record in flight.record_list:
            try:
                print('Writing {}...'.format(record))
                record_ = copy.deepcopy(record)  # stop memory leak
                record_.set_geolocation(flight)
                nc = FLIR01A(record_)
                if overwrite:
                    nc.write()
                else:
                    if os.path.exists(nc.file_path):
                        nc.write()
                    else:
                        print('{} Exists; skipping ...'.format(record_))
            except Exception as e:
                logging.error('Failed netcdf write: {}'.format(record_))
                print('Failed {}...'.format(record_))
                print(e)




'''
----------------------------------
Data Management
----------------------------------
'''


def init_database():
    metadata.create_all()

def write_database():
    metadata.drop_all()
    metadata.create_all()
    update_database()

def update_database():
    for n in range(2):  # Second pass to find all record-flights
        update_flights()
        update_records()

def update_flights():
    """

    Update flights from local DADS .nc files.
    """
    flir_ = Flir()
    for flight in flir_.dad_file_list:
        with session_scope() as session:
            try:
                print(flight)
                flight_obj = Flight(flight)
                session.merge(flight_obj)
                session.commit()
            except Exception as e:
                logging.error('Failed Flight update: {}'.format(flight))
                logging.error(e)
                print(Exception('error for : {}'.format(flight)))
                print(e)
        print(flight)

def update_records():
    """

    Update records from local files.
    """
    flir_ = Flir()
    for record in flir_.record_file_list:
        with session_scope() as session:
            try:
                record_obj = Record(record)
                session.merge(record_obj)
                session.commit()
                print('Record added: {}'.format(record_obj))
            except Exception as e:
                logging.error('Failed Record update: {}'.format(record))
                logging.error(e)
                print(Exception('error for : {}'.format(record)))
                print(e)
        print(record)

def main():
    if opt.write_database:
        write_database()
    if opt.update_database:
        update_database()
    if opt.netcdf:
        if opt.product == 'L1A':
            write_FLIR01A(overwrite=opt.overwrite, year=opt.year, month=opt.month, day=opt.day)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-u", "--update", dest="update_database", action='store_true',
                      help="Update the sqlite3 database", default=False)
    parser.add_option("-n", "--netcdf", dest="netcdf", action='store_true', default=False,
                      help="write netcdf files")
    parser.add_option('-w', '--write_database', dest='write_database', default=False,
                      action='store_true', help="Write database from files, will delete all")
    parser.add_option('-p', '--product', dest='product', default='L1A',
                      help="Product name, default is FLIR01A")
    parser.add_option('-o', '--overwrite', dest='overwrite', default=False,
                      action='store_true', help='Overwite netcdf products, default is update if not-exist')
    parser.add_option('-Y', '--year', dest='year', default=None,
                      type=int, help='define year to write')
    parser.add_option('-M', '--month', dest='month', default=None,
                      type=int, help='Restrict month to write, (year must be set)')
    parser.add_option('-D', '--day', dest='day', default=None,
                      type=int, help='Restrict day to write, (year+month must be set)')

    (opt, args) = parser.parse_args()

    sys.exit(main())