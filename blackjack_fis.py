"""
Sistema de inferencia difusa (Mamdani) para Blackjack-v1 (Gymnasium / Toy Text).

  python blackjack_fis.py --demo-inferencia
  python blackjack_fis.py --vivo 5 --render
  python blackjack_fis.py --eval 5000
  python blackjack_fis.py --eval 100 --render
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass(frozen=True)
class Observacion:
    suma_jugador: int
    carta_crupier: int
    as_utilizable: int

    @classmethod
    def desde_tupla(cls, obs: tuple[int, int, int]) -> "Observacion":
        return cls(int(obs[0]), int(obs[1]), int(obs[2]))


@dataclass
class EstadisticasFIS:
    """Estadisticas acumuladas del controlador difuso."""

    decisiones_hit: int = 0
    decisiones_stick: int = 0
    sin_salida_difusa: int = 0
    centroides: list[float] = field(default_factory=list)
    centroides_hit: list[float] = field(default_factory=list)
    centroides_stick: list[float] = field(default_factory=list)
    pasos_totales: int = 0
    episodios: int = 0
    victorias: int = 0
    derrotas: int = 0
    empates: int = 0
    recompensas: list[float] = field(default_factory=list)

    def registrar_paso(self, accion: int, centroide: float, sin_salida: bool) -> None:
        self.pasos_totales += 1
        if sin_salida:
            self.sin_salida_difusa += 1
        else:
            self.centroides.append(centroide)
        if accion == 1:
            self.decisiones_hit += 1
            if not sin_salida:
                self.centroides_hit.append(centroide)
        else:
            self.decisiones_stick += 1
            if not sin_salida:
                self.centroides_stick.append(centroide)

    def registrar_episodio(self, recompensa: float) -> None:
        self.episodios += 1
        self.recompensas.append(recompensa)
        if recompensa > 0:
            self.victorias += 1
        elif recompensa < 0:
            self.derrotas += 1
        else:
            self.empates += 1

    def imprimir(self) -> None:
        n = self.episodios
        total_dec = self.decisiones_hit + self.decisiones_stick
        print("\n=== Estadisticas del juego ===")
        print(f"  episodios: {n}")
        if n:
            print(f"  recompensa_media: {np.mean(self.recompensas):.4f}")
            print(f"  victorias: {self.victorias} ({100 * self.victorias / n:.1f}%)")
            print(f"  derrotas: {self.derrotas} ({100 * self.derrotas / n:.1f}%)")
            print(f"  empates: {self.empates} ({100 * self.empates / n:.1f}%)")
            print(f"  pasos_promedio_por_episodio: {self.pasos_totales / n:.2f}")

        print("\n=== Estadisticas del FIS (difuso) ===")
        print(f"  decisiones HIT (pedir): {self.decisiones_hit}")
        print(f"  decisiones STICK (plantarse): {self.decisiones_stick}")
        if total_dec:
            print(f"  % HIT: {100 * self.decisiones_hit / total_dec:.1f}%")
            print(f"  % STICK: {100 * self.decisiones_stick / total_dec:.1f}%")
        print(f"  pasos sin salida difusa (fallback): {self.sin_salida_difusa}")

        if self.centroides:
            arr = np.array(self.centroides)
            print(f"  centroide (inclinacion_pedir) - promedio: {arr.mean():.4f}")
            print(f"  centroide - min: {arr.min():.4f}  max: {arr.max():.4f}")
        if self.centroides_hit:
            print(f"  centroide cuando HIT: {np.mean(self.centroides_hit):.4f}")
        if self.centroides_stick:
            print(f"  centroide cuando STICK: {np.mean(self.centroides_stick):.4f}")
        print("  defuzzificacion: centroide | umbral accion: 0.5")


def construir_sistema() -> ctrl.ControlSystemSimulation:
    suma = ctrl.Antecedent(np.arange(4, 22, 1), "suma_jugador")
    crupier = ctrl.Antecedent(np.arange(1, 11, 1), "carta_crupier")
    as_util = ctrl.Antecedent([0, 1], "as_utilizable")
    inclinacion = ctrl.Consequent(np.arange(0, 1.01, 0.01), "inclinacion_pedir")

    suma["hasta_11"] = fuzz.trapmf(suma.universe, [4, 4, 11, 11])
    suma["es_12"] = fuzz.trimf(suma.universe, [12, 12, 12])
    suma["de_13_a_16"] = fuzz.trapmf(suma.universe, [12, 13, 16, 16])
    suma["desde_17"] = fuzz.trapmf(suma.universe, [17, 17, 21, 21])
    suma["soft_18"] = fuzz.trimf(suma.universe, [18, 18, 18])
    suma["soft_19_mas"] = fuzz.trapmf(suma.universe, [19, 19, 21, 21])
    suma["soft_hasta_17"] = fuzz.trapmf(suma.universe, [12, 12, 17, 17])
    suma["soft_baja"] = fuzz.trapmf(suma.universe, [4, 4, 11, 11])

    crupier["d_2_3"] = fuzz.trapmf(crupier.universe, [2, 2, 3, 3])
    crupier["d_4_5_6"] = fuzz.trapmf(crupier.universe, [4, 4, 6, 6])
    crupier["d_2_a_6"] = fuzz.trapmf(crupier.universe, [2, 2, 6, 6])
    crupier["d_7_a_10"] = fuzz.trapmf(crupier.universe, [7, 7, 10, 10])
    crupier["d_9_10_A"] = fuzz.trapmf(crupier.universe, [9, 9, 10, 10])
    crupier["d_2_7_8"] = fuzz.trapmf(crupier.universe, [2, 2, 8, 8])
    crupier["as_visible"] = fuzz.trimf(crupier.universe, [1, 1, 1])

    as_util["no"] = fuzz.trimf(as_util.universe, [0, 0, 0])
    as_util["si"] = fuzz.trimf(as_util.universe, [1, 1, 1])

    inclinacion["plantarse"] = fuzz.trimf(inclinacion.universe, [0.0, 0.0, 0.35])
    inclinacion["pedir"] = fuzz.trimf(inclinacion.universe, [0.65, 1.0, 1.0])

    duro = as_util["no"]
    blando = as_util["si"]

    reglas = [
        ctrl.Rule(duro & suma["hasta_11"], inclinacion["pedir"], label="HD1"),
        ctrl.Rule(duro & suma["desde_17"], inclinacion["plantarse"], label="HD2"),
        ctrl.Rule(duro & suma["es_12"] & crupier["d_4_5_6"], inclinacion["plantarse"], label="HD3"),
        ctrl.Rule(duro & suma["es_12"] & crupier["d_2_3"], inclinacion["pedir"], label="HD4"),
        ctrl.Rule(duro & suma["es_12"] & crupier["d_7_a_10"], inclinacion["pedir"], label="HD5"),
        ctrl.Rule(duro & suma["es_12"] & crupier["as_visible"], inclinacion["pedir"], label="HD6"),
        ctrl.Rule(duro & suma["de_13_a_16"] & crupier["d_2_a_6"], inclinacion["plantarse"], label="HD7"),
        ctrl.Rule(duro & suma["de_13_a_16"] & crupier["d_7_a_10"], inclinacion["pedir"], label="HD8"),
        ctrl.Rule(duro & suma["de_13_a_16"] & crupier["as_visible"], inclinacion["pedir"], label="HD9"),
        ctrl.Rule(blando & suma["soft_baja"], inclinacion["pedir"], label="BL0"),
        ctrl.Rule(blando & suma["soft_hasta_17"], inclinacion["pedir"], label="BL1"),
        ctrl.Rule(blando & suma["soft_19_mas"], inclinacion["plantarse"], label="BL2"),
        ctrl.Rule(blando & suma["soft_18"] & crupier["d_2_7_8"], inclinacion["plantarse"], label="BL3"),
        ctrl.Rule(blando & suma["soft_18"] & crupier["d_4_5_6"], inclinacion["pedir"], label="BL4"),
        ctrl.Rule(blando & suma["soft_18"] & crupier["d_9_10_A"], inclinacion["pedir"], label="BL5"),
        ctrl.Rule(blando & suma["soft_18"] & crupier["as_visible"], inclinacion["pedir"], label="BL6"),
    ]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(reglas))


def inferir_accion(
    sim: ctrl.ControlSystemSimulation,
    obs: Observacion,
    umbral: float = 0.5,
) -> tuple[int, float, bool]:
    """Retorna (accion, centroide, sin_salida_difusa)."""
    sim.reset()
    sim.input["suma_jugador"] = obs.suma_jugador
    sim.input["carta_crupier"] = obs.carta_crupier
    sim.input["as_utilizable"] = obs.as_utilizable
    sim.compute()

    if "inclinacion_pedir" not in sim.output:
        return 0, 0.0, True

    valor = float(sim.output["inclinacion_pedir"])
    return (1 if valor >= umbral else 0), valor, False


def _grados_entrada(sim: ctrl.ControlSystemSimulation, obs: Observacion) -> dict:
    entradas = {
        "suma_jugador": obs.suma_jugador,
        "carta_crupier": obs.carta_crupier,
        "as_utilizable": obs.as_utilizable,
    }
    antecedentes = {a.label: a for a in sim.ctrl.antecedents}
    grados: dict[str, dict[str, float]] = {}
    for nombre, valor in entradas.items():
        ant = antecedentes[nombre]
        grados[nombre] = {
            t: float(fuzz.interp_membership(ant.universe, term.mf, valor))
            for t, term in ant.terms.items()
        }
    return grados


def _crear_entorno(render: bool) -> gym.Env:
    if render:
        try:
            return gym.make("Blackjack-v1", render_mode="human")
        except Exception as e:
            print(
                "\nNo se pudo abrir la ventana pygame. Instala: pip install pygame\n"
                f"Error: {e}\n"
            )
            raise SystemExit(1) from e
    return gym.make("Blackjack-v1")


def _ejecutar_episodio(
    env: gym.Env,
    sim: ctrl.ControlSystemSimulation,
    stats: EstadisticasFIS,
    *,
    render: bool,
    pausa: float,
    consola: bool,
    semilla: int | None = None,
    ep_num: int | None = None,
) -> float:
    carta = {1: "As", **{i: str(i) for i in range(2, 11)}}
    obs, _ = env.reset(seed=semilla)
    if render:
        env.render()

    terminado = False
    recompensa_final = 0.0
    paso = 0

    if consola and ep_num is not None:
        print(f"--- Episodio {ep_num} ---")

    while not terminado:
        o = Observacion.desde_tupla(obs)
        accion, centroide, sin_salida = inferir_accion(sim, o)
        stats.registrar_paso(accion, centroide, sin_salida)

        if consola:
            print(
                f"  paso {paso}: suma={o.suma_jugador}, crupier={carta[o.carta_crupier]}, "
                f"as_util={o.as_utilizable} | centroide={centroide:.3f} -> "
                f"{'HIT' if accion else 'STICK'}"
            )

        obs, recompensa_final, terminado, truncado, _ = env.step(accion)
        terminado = terminado or truncado
        paso += 1

        if render:
            env.render()
            time.sleep(pausa)

    stats.registrar_episodio(recompensa_final)

    if consola:
        res = {1.0: "GANO", -1.0: "PERDIO", 0.0: "EMPATE"}.get(
            recompensa_final, str(recompensa_final)
        )
        print(f"  resultado: {res} (recompensa={recompensa_final})\n")

    return recompensa_final


def demo_inferencia() -> None:
    sim = construir_sistema()
    casos = [
        ("HD1: mano dura, suma baja -> pedir", Observacion(7, 10, 0)),
        ("HD3: suma 12, crupier 4-6 -> plantarse", Observacion(12, 5, 0)),
    ]

    print("\n=== Prueba de inferencia (1-2 reglas) ===")
    print("Mamdani | Defuzzificacion: centroide | Umbral: 0.5\n")

    for titulo, obs in casos:
        grados = _grados_entrada(sim, obs)
        accion, valor, _ = inferir_accion(sim, obs)

        print(titulo)
        print(f"  suma={obs.suma_jugador}, crupier={obs.carta_crupier}, as_util={obs.as_utilizable}")
        for var, terminos in grados.items():
            activos = {k: round(v, 3) for k, v in terminos.items() if v > 0.01}
            print(f"  {var}: {activos}")
        print(f"  centroide (inclinacion_pedir) = {valor:.4f}")
        print(f"  accion: {'HIT (1)' if accion == 1 else 'STICK (0)'}\n")


def evaluar(
    episodios: int = 5000,
    semilla: int = 42,
    render: bool = False,
    render_max: int = 10,
    pausa: float = 0.8,
) -> None:
    stats = EstadisticasFIS()
    sim = construir_sistema()
    rng = np.random.default_rng(semilla)

    if render and episodios > render_max:
        print(
            f"\nCon --render se muestran solo los primeros {render_max} episodios en pygame; "
            f"el resto ({episodios - render_max}) corre sin ventana.\n"
        )

    for i in range(episodios):
        usar_render = render and i < render_max
        if usar_render:
            env = _crear_entorno(True)
        else:
            env = _crear_entorno(False)

        seed = int(rng.integers(0, 2**31))
        _ejecutar_episodio(
            env,
            sim,
            stats,
            render=usar_render,
            pausa=pausa,
            consola=False,
            semilla=seed,
        )
        env.close()

    stats.imprimir()


def corrida_en_vivo(
    episodios: int = 5,
    pausa: float = 0.8,
    render: bool = False,
) -> None:
    stats = EstadisticasFIS()
    sim = construir_sistema()
    env = _crear_entorno(render)

    print("\n=== Corrida en vivo (FIS difuso) ===")
    if render:
        print("(ventana pygame activa)\n")
    else:
        print("(solo consola; usa --render para ver pygame)\n")

    for ep in range(1, episodios + 1):
        _ejecutar_episodio(
            env,
            sim,
            stats,
            render=render,
            pausa=pausa,
            consola=True,
            ep_num=ep,
        )

    env.close()
    stats.imprimir()


def main() -> None:
    parser = argparse.ArgumentParser(description="FIS Mamdani para Blackjack-v1")
    parser.add_argument("--demo-inferencia", action="store_true", help="Prueba de 1-2 reglas")
    parser.add_argument("--eval", type=int, default=0, metavar="N", help="Evaluar N episodios")
    parser.add_argument("--vivo", type=int, default=0, metavar="N", help="Corrida en consola")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Ventana pygame (con --eval solo los primeros episodios)",
    )
    parser.add_argument(
        "--render-max",
        type=int,
        default=10,
        metavar="K",
        help="Cuantos episodios mostrar con pygame en --eval (default 10)",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=0.8,
        help="Segundos entre frames con --render (default 0.8)",
    )
    args = parser.parse_args()

    if args.demo_inferencia:
        demo_inferencia()
    if args.eval > 0:
        evaluar(
            episodios=args.eval,
            render=args.render,
            render_max=args.render_max,
            pausa=args.pausa,
        )
    if args.vivo > 0:
        corrida_en_vivo(episodios=args.vivo, pausa=args.pausa, render=args.render)

    if not (args.demo_inferencia or args.eval > 0 or args.vivo > 0):
        parser.print_help()


if __name__ == "__main__":
    main()
