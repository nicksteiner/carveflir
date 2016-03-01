__author__ = 'nsteiner'

import flir
import random
import pdb
from collections import OrderedDict


def test_init_database():
    flir.init_database()


def test_create_session():
    with flir.session_scope() as session:
        return


def test_get_flir():
    with flir.session_scope() as session:
        return flir.Flir(session)


def test_file_list():
    # test finding files
    with flir.session_scope() as session:
        flir_ = flir.Flir(session)
        assert len(flir_.dad_file_list) > 0

def test_create_flight():
    with flir.session_scope() as session:
        flir_ = flir.Flir(session)
        assert len(flir_.dad_file_list) > 0
        pos = random.randint(0, len(flir_.dad_file_list) - 1)
        return flir.Flight(flir_.dad_file_list[pos])

def test_create_record():
    with flir.session_scope() as session:
        flir_ = flir.Flir(session)
        assert len(flir_.record_file_list) > 0
        pos = random.randint(0, len(flir_.record_file_list) - 1)
        return flir.Record(flir_.record_file_list[pos], False)

def test_record_load():
    with flir.session_scope() as session:
        flir_ = flir.Flir(session)
        assert len(flir_.record_file_list) > 0
        pos = random.randint(0, len(flir_.record_file_list) - 1)
        return flir.Record(flir_.record_file_list[pos], True)

def test_update_records():
    flir.update_flights()
    flir.update_records()

def test_get_flight():
    flight_dates = flir.get_all_flightdates()
    pos = random.randint(0, len(flight_dates) - 1)
    flight_date = flight_dates[pos][0]
    print(flight_date)
    flight = flir.get_flight_bydate(flight_date)
    flight.load()
    return flight

def test_get_record():
    flight = test_get_flight()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list) - 1)
    record = record_list[pos]
    record.load_array()
    record.set_index()
    return record

'''

def test_write_shapefile():
    flight = test_get_flight()
    write_shapefile.write_flight_toshape(flight)
'''

def test_set_record_geolocation():
    flight = test_get_flight()
    flight.load()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list) - 1)
    record = record_list[pos]
    record.set_geolocation(flight)
    return record.geolocation


def test_write_record():
    flight = test_get_flight()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list) - 1)
    record = record_list[pos]
    record.set_geolocation(flight)
    nc = flir.FLIR01A(record)
    nc.write()

def fix_bad_records():
    rec_ = 'Rec-000805,Rec-001448,Rec-001449,Rec-002285,Rec-002946,Rec-002950,Rec-003010,Rec-003153,Rec-003154,Rec-003383,Rec-003517,Rec-003518,Rec-003519,Rec-003520,Rec-003521,Rec-003522,Rec-003523,Rec-004028,Rec-004161,Rec-004162,Rec-004187,Rec-004354,Rec-004483,Rec-004569,Rec-004640,Rec-004846,Rec-004847,Rec-004848,Rec-004890,Rec-005028,Rec-005174,Rec-005175,Rec-005176,Rec-005231,Rec-005232'
    info_ = OrderedDict()
    for record_str in rec_.split(','):
        try:
            record_ = flir.get_record(record_str)
            assert record_ is not None
        except:
            flir.logging.error('{}: Record not found in database.'.format(record_str))
            info_[record_str] = (False, 'Record not found in database.')
            continue
        flight_ = flir.get_flight_byflightid(record_.FlightID)
        flight_.load()
        try:
            record_.set_geolocation(flight_)
            flir.logging.info('{}: PASSING, Time error corrected'.format(record_str))
            info_[record_str] = (True, 'Time error corrected.')
        except Exception as e:
            print(e)
            flir.logging.error('{}: FAILING, {}.'.format(record_str, e))
            info_[record_str] = (False, e)

    with open('./bad_records.txt', 'w') as f:
        f.write('Record,Status,Message\n')
        for rec_str, (status, msg) in info_.iteritems():
            status_str = {True: 'WORKING', False:'FAILING'}[status]
            f.write('{},{},{}\n'.format(rec_str, status_str, msg))



if __name__ == '__main__':
    fix_bad_records()
    print('test 1')
    test_init_database()
    print('test 2')
    test_create_session()
    print('test 3')
    test_get_flir()
    print('test 4')
    test_file_list()
    print('test 5')
    test_create_flight()
    print('test 6')
    test_create_record()
    print('test 7')
    test_record_load()
    print('test 8')
    test_update_records()
    print('test 9')
    test_get_flight()
    print('test 10')
    test_get_record()
    print('test 11')
    test_set_record_geolocation()
    print('test 12')
    #test_write_record()
    print('done')

	

