from embedia.utils.binary_helper import BinaryGlobalMask
from pathlib import Path
import re

def declare_array(dt_type, var_name, dt_conv, data_array, limit=80):

    if dt_conv is None:
        dt_conv = lambda x:x
    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            line += f'''{dt_conv(v)}, '''
            if len(line) >= limit:
                val += line + '\n    '
                line = ''
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code


def declare_array2(toti,xBits,lista_contadores,dt_type, var_name, dt_conv, data_array, limit=80):

    if data_array is None:
        val = 'NULL'
    else:
        val = ''
        line = ''
        for v in data_array:
            
            lista_contadores[2] = lista_contadores[2] + 1
            
            if xBits==16:
                if v == 1.0:  
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_16())[lista_contadores[1]]
            elif xBits==32:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_32())[lista_contadores[1]]
            elif xBits==64:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_64())[lista_contadores[1]]
            else:
                if v == 1.0: 
                    lista_contadores[0] += (BinaryGlobalMask.get_mask_8())[lista_contadores[1]]
            
            if lista_contadores[1] == xBits-1 or (lista_contadores[2] == toti):
                
                line += f'''{dt_conv(lista_contadores[0])}, '''
                if len(line) >= limit:
                    val += line + '\n    '
                    line = ''
                lista_contadores[1] = 0
                lista_contadores[0] = 0
                
            else:
                lista_contadores[1] = lista_contadores[1] +1
            
            
        if line != '':
            val += line
        val = val[:-2]  # remove last comma and space

    if var_name is None or var_name == '':
        code = ''  # only values
    else:
        code = f'{dt_type} {var_name}[] ='
    code += f'''{{
    {val}
    }}'''

    return code



def replace_c_define(content, values):
    """
    Reemplaza valores de constantes definidas con #define en código C/C++.

    Args:
        content (str): El contenido del código fuente donde buscar los #define
        values: Puede ser:
            - Una tupla (nombre_constante, valor_constante)
            - Una lista de tuplas [(nombre1, valor1), (nombre2, valor2), ...]
            Los valores pueden ser: int, float, str, bool

    Returns:
        str: El contenido con los valores de #define reemplazados

    Examples:
        >>> code = "#define PI 3.14\\n#define MAX_SIZE 100"
        >>> replace_c_define(code, ("PI", 3.14159))  # float
        '#define PI 3.14159\\n#define MAX_SIZE 100'

        >>> replace_c_define(code, [("PI", 3.14159), ("MAX_SIZE", 200)])  # mixed types
        '#define PI 3.14159\\n#define MAX_SIZE 200'

        >>> replace_c_define(code, ("DEBUG", True))  # bool -> "True"
        >>> replace_c_define(code, ("FLAG", False))  # bool -> "False"
    """

    # Normalizar la entrada: convertir tupla individual a lista
    if isinstance(values, tuple):
        values = [values]

    # Crear una copia del contenido para modificar
    result = content

    # Procesar cada constante
    for name, value in values:
        # Patrón regex para encontrar #define NOMBRE valor
        # Captura espacios/tabs después de #define y después del nombre
        pattern = r'(#define\s+' + re.escape(name) + r'\s+)([^\s\n]*)'

        # Reemplazar manteniendo la estructura original
        replacement = r'\g<1>' + str(value)
        result = re.sub(pattern, replacement, result)

    return result


class BlockContext:
    """Context manager for code blocks."""

    def __init__(self, builder, header, footer="}"):
        self.builder = builder
        self.header = header
        self.footer = footer

    def __enter__(self):
        self.builder.add(self.header).inc()
        return self.builder  # Retorna el builder para usar sus métodos

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.builder.dec().add(self.footer)
        return False


class CBuilder:
    """Text builder with indentation support using simplified method names."""

    def __init__(self, indent_size=4):
        self.indent_size = indent_size
        self._current_indent = 0
        self._lines = []

    def add(self, text=""):
        """Add text with current indentation."""
        if text:
            indent = ' ' * self._current_indent
            self._lines.append(indent + text.replace('\n', '\n' + indent))
        return self

    def append(self,text=""):
        """Add text with current indentation."""
        self.add(text)
        return self

    def inc(self):
        """Increase indentation level."""
        self._current_indent += self.indent_size
        return self

    def dec(self):
        """Decrease indentation level."""
        self._current_indent = max(0, self._current_indent - self.indent_size)
        return self

    def bgn(self, header, footer="}"):
        """Context manager for blocks with automatic indentation."""
        return BlockContext(self, header, footer)

    def end(self, footer=""):
        """Close a block with optional footer."""
        self.dec()
        if footer:
            self.add(footer)
        return self

    def get_code(self):
        """Return the built text as a single string."""
        return '\n'.join(self._lines)

    def __str__(self):
        return self.get_code()

    def clear(self):
        """Reset the builder."""
        self._lines = []
        self._current_indent = 0
        return self

    def to_array(self, values, sep=', ', fmt=''):
        formatted_values = []
        for x in values:
            try:
                formatted_values.append(f"{x: {fmt}}")
            except (ValueError, TypeError):
                formatted_values.append(str(x))
        return sep.join(formatted_values)

    def indent_text(self, text, times=1, char=' '):
        """apply text with indentation."""
        indent_size = self.indent_size * times
        indent_block = char * indent_size
        return indent_block + text.replace('\n', '\n' + indent_block)

    def printf(self, format_string, *args):
        """Add a printf statement with C formatting."""
        arg_str = ', '.join(str(arg) for arg in args)
        if args:
            self.add(f'printf("{format_string}", {arg_str});')
        else:
            self.add(f'printf("{format_string}");')
        return self

    def load(self, filename):
        """Load content from file and replace current content."""
        path = Path(filename)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            self.clear()
            # Split by lines and add without extra indentation since load replaces content
            for line in content.splitlines():
                self._lines.append(line)
        else:
            raise FileNotFoundError(f"File '{filename}' not found")
        return self

    def save(self, filename):
        """Save current content to file."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.get_code(), encoding='utf-8')
        return self


class ArduinoBuilder(CBuilder):
    """Arduino-specific code builder with Serial.print printf implementation."""

    def printf(self, format_string, *args):
        """Add printf-like statements using Serial.print for Arduino."""
        # Improved pattern to capture:
        # % - (optional) flags (-, +, space, 0)
        # (optional) width
        # (optional) .precision
        # conversion type
        pattern = r'%([-+ 0]*)(\d*)(?:\.(\d+))?([dfsuoxX])'

        matches = list(re.finditer(pattern, format_string))
        if len(matches) != len(args):
            raise ValueError(f"Number of format specifiers ({len(matches)}) doesn't match arguments ({len(args)})")

        last_end = 0

        for i, match in enumerate(matches):
            # Add text before the specifier
            if match.start() > last_end:
                text = format_string[last_end:match.start()]
                self.append(f'Serial.print("{text}");')

            # Process the specifier
            flags = match.group(1)
            width = match.group(2)
            precision = match.group(3)
            type_spec = match.group(4)
            arg = args[i]

            # Debug print
            #print(f"Processing: {match.group(0)}")
            #print(f"Flags: '{flags}', Width: '{width}', Precision: '{precision}', Type: '{type_spec}'")

            if type_spec == 's':
                # String handling (width for strings would need manual padding in Arduino)
                self.append(f'Serial.print("{arg}");')
            elif type_spec == 'f':
                # Float handling
                decimals = precision if precision else '2'
                self.append(f'Serial.print({arg}, {decimals});')
            elif type_spec in ('d', 'i', 'u', 'o', 'x', 'X'):
                # Integer types (width and zero-padding would need manual handling)
                self.append(f'Serial.print({arg});')
            else:
                self.append(f'Serial.print({arg});')

            last_end = match.end()

        # Add remaining text
        if last_end < len(format_string):
            text = format_string[last_end:]
            self.append(f'Serial.print("{text}");')





