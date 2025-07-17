# AEIC Requirements Specification

*Version:* 0.1
*Date:* 2025-07-17

---

## 1. Framework

![Framework Diagram](./assets/AEIC_framework.png)

---

## 2. Inputs

| Source                               | Format                                      | Mandatory?                            | Key Fields / Sections                                                                                   |
| ------------------------------------ | ------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Performance model file              | **TOML** <br>`data/PerformanceModel/*.toml` | ✔                                     | `General_Information`, `Speeds`, `LTO_performance`, `flight_performance`                                |
| Missions / OAG Schedule                    | **SQL**               | ✔                                     | `dep_airport`, `arr_airport`, `dep_datetime`, `arr_datetime`, `great_circle_distance?`, `ac_code`, `load_factor?` |
| Airports database                    | **SQL**               | ✔                                     | `iata_code`, `lat`, `lon`, `altitude`                                                               |
| Configuration File              | **TOML** (`default_config.toml`)            | ✔                                     | `General Information`, `LTO data`, `Missions`, `Emissions`, `Output`                                    |
| Weather data <br>(ERA‑5, MERRA‑2 …) | **NetCDF‑4**                     | ✖ (falls back to standard atmosphere) | Winds: **u**, **v**, `relative_humidity`                                    |

> **?** means the field is optional

---

## 3. Modules & Intermediates

### 3.1 Trajectory Module

*For each mission* a *`Ntot`‑length* (total number of mission points) table **`trajectory_<FLIGHT_ID>`** is saved with columns:

| Column name                          | Description                  | Shape           |
| ----------------------------------------- | --------------------------- | --------------- |
| `fuelFlow`                        | Total fuel flow rate [`kg/s`] | (Ntot,) |
| `acMass`                   | Total mass of aircraft over the trajectory [`kg`]       | (Ntot,) |
| `fuelMass` | Mass of fuel over the trajectory [`kg`]    | (Ntot,)    |
| `groundDist` | Cumulative ground distance covered [`m`]    | (Ntot,)    |
| `altitude` | Altitude [`m`]    | (Ntot,)    |
| `FLs` | Altitude in flight levels [ ]    | (Ntot,)    |
| `rocs` | Rate of climb (negative for rate of descent) [`m/s`]    | (Ntot,)    |
| `flightTime` | Elapsed flight time [`s`]    | (Ntot,)    |
| `latitude` | Latitude at mission point [`degrees`]    | (Ntot,)    |
| `longitude` | Longitude at mission point [`degrees`]    | (Ntot,)    |
| `azimuth` | Azimuth to arrival airport [`degrees`]    | (Ntot,)    |
| `heading` | Heading at mission point  [`degrees`]    | (Ntot,)    |
| `tas` | True airspeed  [`m/s`]    | (Ntot,)    |
| `groundSpeed` | Ground speed  [`m/s`]    | (Ntot,)    |
| `FL_weight` | weighting used in linear interpolation over flight levels  [ ]    | (Ntot,)    |

### 3.2 Weather Module

The trajectory table is extended in‑place with:

| Field               | Unit  |
| ------------------- | ----- |
| `wind_u`, `wind_v`  | m s⁻¹ |
| `relative_humidity` |       |

### 3.3 Emissions Module

For each mission the module stores **`emissions_<FLIGHT_ID>`** containing:

| Column name                          | Description                  | Shape           |
| ----------------------------------------- | --------------------------- | --------------- |
| `emission_indices`                        | EI by species at each point in trajectory | (Ntot, species) |
| `pointwise_emissions_g`                   | Emissions in grams at each point in trajectory       | (Ntot, species) |
| `LTO_emission_indices`, `LTO_emissions_g` | LTO emission indices or emission in grams    | (4, species)    |
| `APU_emission_indices`, `APU_emissions_g` | APU emission indices or emission in grams              | (1, species)    |
| `GSE_emissions_g`                         | Ground‑support equipment    | (1, species)    |
| `summed_emission_g`                       | Flight total summed emissions (trajectory+LTO+APU+GSE+Lifecycle)                | (species,)      |

Each of the above data points have columns (`n` = number of points):

| Column name                           | Shape           |
| --------------------------- | --------------- |
| `CO2`                        | (n,) |
| `HC`                        | (n,) |
| `CO`                        | (n,) |
| `NOx`                        | (n,) |
| `NO`                        | (n,) |
| `NO2`                        | (n,) |
| `HONO`                        | (n,) |
| `PMnvol`                        | (n,) |
| `PMnvol_lo`                        | (n,) |
| `PMnvol_hi`                        | (n,) |
| `PMnvolN`                        | (n,) |
| `PMnvolN_lo`                        | (n,) |
| `PMnvolN_hi`                        | (n,) |
| `PMnvolN_GMD`                        | (n,) |
| `PMvol`                        | (n,) |
| `OCic`                        | (n,) |
| `SO2`                        | (n,) |
| `SO4`                        | (n,) |

The emissions module also calculates a total fuel burn from trajectory, APU, GSE and LTO.

> **NOTE:** $CO_2$, $H_2O$ and $SO_2$, $SO_4$ are scalar products of fuel burn and <br> need not be saved in intermediate outputs since they can be computed easily.

### 4. Output Module

| ID     | Artifact               | Format                                                                                   | Granularity                     |
| ------ | ---------------------- | ---------------------------------------------------------------------------------------- | ------------------------------- |
| **O1** | Emissions by flight/aircraft type |**NetCDF‑4** (`by_flight_emissions_YYYYMM.nc`)                                                                            | 1 row per flight                |
| **O2** | Total summed emissions         | **NetCDF‑4** (`summed_emissions_YYYYMM.nc`)                                                                              | (species, total\_kg) |
| Gridded emissions  | **NetCDF‑4** (`gridded_emissions_YYYYMM.nc`)                                                     | lon × lat × alt × time x species      |
| Model‑specific exports | Files as required by <br>**GEOS‑Chem**, **ACAI**, other AQ/climate models | varies                          |

---

## 5. Nomenclature

| Term               | Meaning                                               |
| ------------------ | ----------------------------------------------------- |
| **LTO**            | Landing & Take‑Off cycle (≤ 3000 ft)                  |
| **APU**            | Auxiliary Power Unit                                  |
| **GSE**            | Ground Support Equipment                              |
| **EI**             | Emission Index, g pollutant per kg fuel               |
