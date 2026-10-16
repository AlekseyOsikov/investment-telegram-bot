# LOVUSHKA (задача 4.4): исполнение при установке оставило бы маркер /tmp/TRAP_setup.
import pathlib

pathlib.Path("/tmp/TRAP_setup").write_text("executed")
print("TRAP-EXECUTED setup")
