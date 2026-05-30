"""Monitoreo operativo: tiempo, CPU, memoria y latencia."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import psutil

from src.utils import guardar_json, timestamp_archivo


@dataclass
class MonitorOperativo:
    """Registra métricas computacionales simples de una ejecución."""

    nombre: str
    inicio: float = field(default_factory=time.time)
    cpu_inicio: float = field(default_factory=lambda: psutil.cpu_percent(interval=0.1))
    memoria_inicio_mb: float = field(default_factory=lambda: psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    eventos: list = field(default_factory=list)

    def registrar_evento(self, nombre: str, extra: dict | None = None) -> None:
        """Agrega un evento intermedio al monitoreo."""
        self.eventos.append(
            {
                "evento": nombre,
                "tiempo_segundos": round(time.time() - self.inicio, 3),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memoria_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 3),
                "extra": extra or {},
            }
        )

    def finalizar(self, registros: int | None = None, estado: str = "OK") -> dict:
        """Devuelve resumen final del monitoreo."""
        tiempo_total = time.time() - self.inicio
        memoria_fin = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        return {
            "nombre": self.nombre,
            "estado": estado,
            "tiempo_total_segundos": round(tiempo_total, 3),
            "registros": int(registros) if registros is not None else None,
            "latencia_segundos_por_registro": round(tiempo_total / registros, 8) if registros else None,
            "cpu_inicio_percent": self.cpu_inicio,
            "cpu_fin_percent": psutil.cpu_percent(interval=0.1),
            "memoria_inicio_mb": round(self.memoria_inicio_mb, 3),
            "memoria_fin_mb": round(memoria_fin, 3),
            "eventos": self.eventos,
        }


def guardar_monitoreo(resumen: dict, logs_dir: str, prefijo: str) -> str:
    """Guarda monitoreo en JSON."""
    path = os.path.join(logs_dir, f"monitoring_{prefijo}_{timestamp_archivo()}.json")
    guardar_json(resumen, path)
    return path
