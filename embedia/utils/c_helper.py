from embedia.utils.binary_helper import BinaryGlobalMask

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

    def inc(self):
        """Increase indentation level."""
        self._current_indent += self.indent_size
        return self

    def dec(self):
        """Decrease indentation level."""
        self._current_indent = max(0, self._current_indent - self.indent_size)
        return self

    def new_block(self, header):
        """
        Add a block with automatic indentation.

        Args:
            header: Opening line (e.g., "if (condition) {")
        """
        self.add(header).inc()
        return self  # Just return self for chaining

    def end_block(self, footer=""):
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

    def indent(self, text, times=1, char=' '):
        """apply text with indentation."""
        indent_size = self.indent_size*times
        indent_block = char * indent_size
        return indent_block + text.replace('\n', '\n' + indent_block)



