import csv
import json
import logging
import os
import tempfile
import unittest
from dataclasses import asdict

import torch

from evaluation.report import export_results
from hardware import (C3HardwareConfig, ConvHardwareConfig, HardwareMetrics,
                      calculate_c3_metrics, calculate_conv_metrics, load_config)
from hardware.components import (GROUPS, EvaluationContext, default_specs, register_model,
                                 resolve_components, number)


def extra(name="extra", group="input_periphery", **overrides):
    spec = dict(name=name, model="fixed_current", group=group,
                count={"rule": "one"}, activity={"rule": "one"}, timing=["read"],
                params=dict(supply_v=1., current_ua=2., area_um2=3.))
    spec.update(overrides)
    return spec


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones(1, 96, 1, 1, 1)
        self.w = torch.ones(2, 96, 1, 1)
        self.levels = torch.tensor([-1., 1.])

    def calc(self, c, batch=1, bins=1):
        fn = calculate_conv_metrics if isinstance(c, ConvHardwareConfig) else calculate_c3_metrics
        return fn(c, self.x.repeat(batch, 1, 1, 1, bins), self.w, self.levels, 0)

    def test_c3_hand_formula(self):
        m = self.calc(C3HardwareConfig(active_rows=8))
        energies = dict(column=.000132, column_driver=.0156684, VI=.005832, LIF=.0063624)
        for name, energy in energies.items():
            self.assertAlmostEqual(m.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(m.energy_nj, .0279948, places=12)
        self.assertAlmostEqual(m.latency_us, .482, places=12)
        self.assertAlmostEqual(m.area_mm2, .01025968, places=12)
        self.assertEqual(m.components['column_driver'].count, 4)
        self.assertEqual(m.components['column'].model, 'c3_column')

    def test_conventional_hand_formula_actual_G(self):
        c = ConvHardwareConfig(active_rows=8, ref_sub_curr=3., ref_sub_lat=7., ref_sub_area=2.)
        m = self.calc(c)
        # All weights +1 -> Gmax; reference G(0)=(Gmin+Gmax)/2.
        data_a = 96 * 2 * .0005 * .1
        ref_a = 96 * 2 * (.000005 + .0005) / 2 * .1
        expected = dict(crossbar=data_a * 1.1 * 4.5,
                        reference_array=ref_a * 1.1 * 4.5,
                        DA=1.1 * 6.1 * (2 * 12) * 4.5e-6,
                        reference_subtractor=1.1 * 3 * 2 * 8 * 7e-6,
                        LIF=1.1 * 6 * 2 * (8 * (4.5 + 7) + 2) * 1e-6)
        for name, energy in expected.items():
            self.assertAlmostEqual(m.components[name].energy_nj, energy, places=7)
        self.assertAlmostEqual(m.latency_us, .094)
        self.assertAlmostEqual(m.area_mm2, (4 * 136.67 + 128 * 30.22 + 64 * (86.79 + 2)) * 1e-6)

    def test_override_driver_grouping_only(self):
        a = self.calc(C3HardwareConfig(active_rows=8))
        c = C3HardwareConfig(active_rows=8, components=[dict(name="column_driver",
            count=dict(columns_per_group=7), activity=dict(columns_per_group=7))])
        b = self.calc(c)
        self.assertEqual(b.components['column_driver'].count, 20)  # ceil(64/7) per tile
        self.assertAlmostEqual(b.components['column_driver'].energy_nj,
                               a.components['column_driver'].energy_nj * 5)
        for name in ('column', 'VI', 'LIF'):
            self.assertEqual(a.components[name], b.components[name])
        self.assertEqual(a.read_current_ua, b.read_current_ua)
        self.assertEqual(a.latency_us, b.latency_us)

    def test_new_component_every_group_parallel_latency(self):
        for cls in (ConvHardwareConfig, C3HardwareConfig):
            base = self.calc(cls(active_rows=8))
            c = cls(active_rows=8, components=[extra('extra_' + group, group) for group in GROUPS])
            m = self.calc(c)
            duration = c.xbar_lat if cls is ConvHardwareConfig else c.col_lat
            self.assertAlmostEqual(m.energy_nj - base.energy_nj, 4 * 2 * 8 * duration * 1e-6)
            self.assertAlmostEqual(m.area_mm2 - base.area_mm2, 12e-6)
            self.assertEqual(m.latency_us, base.latency_us)
            self.assertAlmostEqual(sum(g.energy_nj for g in m.groups.values()), m.energy_nj)
            self.assertAlmostEqual(sum(g.area_mm2 for g in m.groups.values()), m.area_mm2)
            self.assertAlmostEqual(sum(g.latency_us for g in m.groups.values()), m.latency_us)

    def test_custom_stage_lif_integrates_no_double_count(self):
        c = C3HardwareConfig(active_rows=8)
        _, schedule = default_specs(c)
        schedule.insert(1, dict(name="buffer", duration_ns=5., repeat="row_phase", owner="buffer"))
        custom = extra("buffer", timing=["buffer"])
        parallel = extra("parallel", timing=["buffer"])
        extended = C3HardwareConfig(active_rows=8, components=[custom, parallel], schedule=schedule)
        a, b = self.calc(c), self.calc(extended)
        self.assertAlmostEqual(b.latency_us - a.latency_us, .040)
        self.assertAlmostEqual(b.components['buffer'].energy_nj, 2 * 5 * 8e-6)
        self.assertEqual(b.components['parallel'].latency_us, 0.)
        self.assertAlmostEqual(b.components['LIF'].energy_nj - a.components['LIF'].energy_nj,
                               1.1 * 6 * 2 * 40e-6)

    def test_count_activity_rules_and_frequencies(self):
        c = C3HardwareConfig(active_rows=8)
        _, schedule = default_specs(c)
        ctx = EvaluationContext(c, 3, 2, [8, 4], 2, 70, schedule)
        installed = dict(one=1, tiles=4, physical_rows=256, physical_columns=256,
                         logical_columns=140, output_bank=128, logical_outputs=70, column_groups=40)
        phased = dict(one=8, tiles=24, physical_rows=1536, physical_columns=1536,
                      logical_columns=840, output_bank=1024, logical_outputs=560, column_groups=240)
        for rule in installed:
            options = dict(rule=rule)
            if rule == 'column_groups':
                options['columns_per_group'] = 7
            self.assertEqual(ctx.instances(options), installed[rule])
            self.assertEqual(ctx.instances(options, phased=True), phased[rule])
        for frequency, events in [('image', 3), ('bin', 6), ('row_phase', 48), ('stage', 48)]:
            spec = extra(activity=dict(rule='one', frequency=frequency))
            self.assertEqual(ctx.powered_ns(spec), 50 * events)

    def test_explicit_installed_and_powered_counts(self):
        c = C3HardwareConfig(active_rows=8, components=[extra(
            count=dict(rule='fixed', value=5),
            activity=dict(rule='fixed', value=3))])
        m = self.calc(c)
        self.assertEqual(m.components['extra'].count, 5)
        self.assertAlmostEqual(m.components['extra'].area_mm2, 15e-6)
        self.assertAlmostEqual(m.components['extra'].energy_nj, 3 * 8 * 50 * 2e-6)
        for value in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                C3HardwareConfig(components=[extra(count=dict(rule='fixed', value=value))])

    def test_batches_normalize_custom_components_and_groups(self):
        for cls in (ConvHardwareConfig, C3HardwareConfig):
            c = cls(active_rows=8, components=[extra('extra_' + g, g) for g in GROUPS])
            one = self.calc(c, bins=2)
            combined = HardwareMetrics()
            combined.add(self.calc(c, batch=2, bins=2))
            combined.add(self.calc(c, batch=1, bins=2))
            normalized = combined.normalize()
            for field in ('energy_nj', 'latency_us', 'area_mm2', 'ops', 'read_current_ua'):
                self.assertAlmostEqual(getattr(normalized, field), getattr(one, field), places=6)
            for name, comp in normalized.components.items():
                self.assertEqual(comp.count, one.components[name].count)
                self.assertEqual(comp.group, one.components[name].group)
                self.assertAlmostEqual(comp.energy_nj, one.components[name].energy_nj, places=6)
            layers = HardwareMetrics()
            layers.add(one, distinct_layer=True)
            layers.add(one, distinct_layer=True)
            self.assertAlmostEqual(layers.area_mm2, 2 * one.area_mm2)
            self.assertAlmostEqual(sum(g.area_mm2 for g in layers.groups.values()), layers.area_mm2)

    def test_examples_and_once_per_image_stage(self):
        examples = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'hardware', 'examples')
        for filename in ('c3_driver_override.json', 'c3_added_stage.json'):
            config = load_config(C3HardwareConfig, os.path.join(examples, filename))
            self.assertGreater(self.calc(config).energy_nj, 0)
        c = C3HardwareConfig(active_rows=8)
        _, schedule = default_specs(c)
        schedule.insert(0, dict(name='setup', duration_ns=11, repeat='image', owner='setup'))
        custom = C3HardwareConfig(active_rows=8, schedule=schedule,
            components=[extra('setup', timing=['setup'])])
        a = self.calc(c, batch=3, bins=2)
        b = self.calc(custom, batch=3, bins=2)
        self.assertAlmostEqual(b.latency_us - a.latency_us, .033)
        self.assertAlmostEqual(b.components['setup'].energy_nj, 2 * 11 * 3e-6)
        self.assertAlmostEqual(b.components['LIF'].energy_nj - a.components['LIF'].energy_nj,
                               1.1 * 6 * 2 * 11 * 3e-6)

    def test_validation(self):
        bad = [dict(vdd=float('nan')), dict(col_curr=-1), dict(xbar_col=2.5),
               dict(driver_part=True), dict(active_rows=65), dict(max_res=1000),
               dict(components={}), dict(components=[extra(model='typo')]),
               dict(components=[extra(group='typo')]), dict(components=[extra(typo=1)]),
               dict(components=[extra(params=dict(supply_v=1, current_ua=1, area_um2=1, typo=1))]),
               dict(components=[extra(count=dict(rule='column_groups', columns_per_group=0))]),
               dict(components=[extra(activity=dict(rule='one', frequency='unknown'))]),
               dict(components=[extra(timing=['unknown'])]), dict(components=[extra(), extra()]),
               dict(schedule=[]), dict(schedule=[dict(name='lif', duration_ns=-1, repeat='bin', owner='LIF')]),
               dict(schedule=[dict(name='lif', duration_ns=1, repeat='row_phase', owner='LIF')]),
               dict(schedule=[dict(name='lif', duration_ns=1, repeat='bin', owner='unknown')])]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                C3HardwareConfig(**params)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(dict(components=[extra(model='missing')]), f)
            f.flush()
            with self.assertRaises(ValueError):
                load_config(C3HardwareConfig, f.name)

    def test_python_evaluator_registration(self):
        name = 'test_charge_model'
        def validator(p):
            if set(p) != {'area_um2', 'energy_per_ns'}:
                raise ValueError('Unknown parameters')
            for key, value in p.items():
                number(value, key)
        register_model(name, lambda s, c, n, powered_ns: s['params']['energy_per_ns'] * powered_ns, validator)
        c = C3HardwareConfig(components=[extra(model=name, params=dict(area_um2=3, energy_per_ns=.1))])
        self.assertAlmostEqual(self.calc(c).components['extra'].energy_nj, 5.)
        with self.assertRaises(ValueError):
            register_model(name, lambda *args: 0, validator)

    def test_export_resolved_schema_and_config_roundtrip(self):
        c = C3HardwareConfig(active_rows=8, components=[extra()])
        self.assertEqual(c, C3HardwareConfig(**json.loads(json.dumps(asdict(c)))))
        m = self.calc(c, batch=2)
        with tempfile.TemporaryDirectory() as folder:
            export_results(folder, 'test', 'test', 'c3cim', c, 0, 2, 1,
                           {'layer': m}, m.normalize(), logging.getLogger('test'))
            with open(os.path.join(folder, 'comparison_summary.csv')) as f:
                row = next(csv.DictReader(f))
            with open(row['result_json']) as f:
                report = json.load(f)
            self.assertEqual(report['schema_version'], 2)
            self.assertEqual(report['configuration'], asdict(c))
            self.assertEqual(report['schedule'], resolve_components(c)[1])
            self.assertEqual(report['resolved_components'], resolve_components(c)[0])
            self.assertEqual(set(report['network']['groups']), set(GROUPS))
            self.assertEqual(report['network']['components']['extra']['group'], 'input_periphery')
            self.assertAlmostEqual(report['network']['energy_nj'], m.energy_nj / 2)


if __name__ == '__main__':
    unittest.main()
