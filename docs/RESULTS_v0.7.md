# model_v07 (conforme a RULES.md)

Cambios respecto a v06:
- Sin constantes de frame: el pre-saque de cada mitad (D desde su primer frame graduado hasta el tag `Kick off (start)` del Smart Tagging) sustituye a la ventana fija `110808:113742`.
- `p_exit_d = 15.0` (valor a priori); ya no se ajusta con el GT público.

Chequeo en el GT público (solo informativo, no se usa para ajustar): macroF1 0.9140 (v06: 0.9149), spellF1 0.5935 (igual), Alive/Dead F1 0.9875 (v06: 0.9880).
La ventana pre-saque de la 2ª parte sale 2903/2934 en D sin ninguna constante.

`model_v04`-`model_v06` contienen la ventana fija y **no deben presentarse**.
