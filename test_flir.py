__author__ = 'nsteiner'

import flir
import random
import pdb


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