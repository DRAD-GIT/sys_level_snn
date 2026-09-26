"""The two original architectures expressed in the modular format.

conventional_cim / c3cim take the parameter objects of hardware/configs.py
(ConvHardwareConfig / C3HardwareConfig, default component set only) and
return equivalent Architectures, so results can be compared one to one.
"""
from cim.architecture import Architecture, Component, Crossbar, Precision, Stage


def conventional_cim(c):
    """Current-mode CIM: analog multi-level cells, optional reference array,
    per-column DA, LIF bank; supply-side read energy vdd * I_cell * t_read."""
    stages = [Stage("read", c.xbar_lat)]
    components = [Component("crossbar", model="crossbar_read", count="tiles", during=["read"],
                            supply_v=c.vdd, area_um2=c.xbar_area, group="crossbar")]
    if c.reference_array:
        stages.append(Stage("subtract", c.ref_sub_lat))
        components.append(Component("reference_array", model="reference_read", count="tiles",
                                    during=["read"], supply_v=c.vdd, area_um2=c.xbar_area,
                                    group="crossbar"))
    components.append(Component("DA", count="physical_columns", on="used_columns", during=["read"],
                                supply_v=c.vdd, static_ua=c.DA_curr, area_um2=c.DA_area,
                                group="output_periphery"))
    if c.reference_array:
        components.append(Component("reference_subtractor", count="output_bank", on="outputs",
                                    during=["subtract"], supply_v=c.vdd, static_ua=c.ref_sub_curr,
                                    area_um2=c.ref_sub_area, group="output_periphery"))
    stages.append(Stage("lif", c.lif_lat, level="timestep"))
    components.append(Component("LIF", count="output_bank", on="outputs", during=["timestep"],
                                supply_v=c.vdd, static_ua=c.lif_curr, area_um2=c.lif_area,
                                group="lif"))
    return Architecture(
        "conventional",
        Crossbar(rows=c.xbar_row, cols=c.xbar_col, r_on=c.min_res, r_off=c.max_res,
                 v_read=c.vread, active_rows=c.active_rows, reference_columns=c.reference_array),
        Precision(weight_encoding="analog"), stages, components)


def c3cim(c):
    """C3CIM macro: fixed column-source current, shared drivers, VI, LIF."""
    groups = {"rule": "column_groups", "size": c.driver_part}
    return Architecture(
        "c3cim",
        Crossbar(rows=c.xbar_row, cols=c.xbar_col, r_on=c.min_res, r_off=c.max_res,
                 active_rows=c.active_rows),
        Precision(weight_encoding="analog"),
        [Stage("read", c.col_lat), Stage("VI", c.VI_lat), Stage("lif", c.lif_lat, level="timestep")],
        [Component("column", count="physical_columns", on="used_columns", during=["read"],
                   supply_v=c.vdd, static_ua=c.col_curr, area_um2=c.col_area, group="crossbar"),
         Component("column_driver", count=groups, on=groups, during=["read"], supply_v=c.vdd,
                   static_ua=c.driver_curr, area_um2=c.driver_area, group="input_periphery"),
         Component("VI", count="physical_columns", on="used_columns", during=["VI"],
                   supply_v=c.VI_supp, static_ua=c.VI_curr, area_um2=c.VI_area,
                   group="output_periphery"),
         Component("LIF", count="output_bank", on="outputs", during=["timestep"], supply_v=c.vdd,
                   static_ua=c.lif_curr, area_um2=c.lif_area, group="lif")])
