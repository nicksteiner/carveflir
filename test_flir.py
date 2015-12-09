__author__ = 'nsteiner'

import flir
import random


def test_get_flir():
    flir.Flir()

def test_get_flight():
    flight_dates = flir.get_all_flightdates()
    pos = random.randint(0, len(flight_dates))
    flight_date = flight_dates[pos][0]
    flight = flir.get_flight_bydate(flight_date)
    flight.load()
    return flight

def test_get_record():
    flight = test_get_flight()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list))
    record = record_list[pos]
    record.load_array()
    record.set_index()
    return record

'''

def test_write_shapefile():
    flight = test_get_flight()
    write_shapefile.write_flight_toshape(flight)
'''

def test_load_array():
    record = test_get_record()
    record.load_array()

def test_set_record_geolocation():
    flight = test_get_flight()
    flight.load()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list))
    record = record_list[pos]
    record.set_geolocation(flight)
    return record.geolocation

def test_write_record():
    flight = test_get_flight()
    record_list = flir.get_records_byflightid(flight.FlightID)
    pos = random.randint(0, len(record_list))
    record = record_list[pos]
    record.set_geolocation(flight)
    nc = flir.FLIR01A(record)
    nc.write()

def test_update_records():
    flir.update_flights()
    flir.update_records()