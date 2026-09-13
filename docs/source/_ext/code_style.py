"""Code-block colours that match VS Code: One Dark Pro Darker (dark), GitHub Light (light).

Pygments' bundled ``one-dark`` paints every name red, and no bundled style follows
the editor, because the editor colours by *role* -- a call, a constant, a keyword
argument -- while Pygments' Python lexer emits one ``Name`` token for all of them.
So there are two parts here:

- ``VSCodePythonLexer`` refines those ``Name`` tokens the way VS Code's Python
  grammar does: a name before ``(`` is a call, an ALL_CAPS name is a constant, a
  name before ``=`` inside a call is a keyword argument.
- ``OneDarkProDarker`` / ``GitHubLight`` colour the refined tokens with the
  themes' own hex values.

Colours are baked in at build time: no JavaScript, nothing changes after the page
loads. What stays out of reach is VS Code's *semantic* colouring (Pylance), which
needs a language server.
"""

from __future__ import annotations

import re

from pygments.lexers.python import PythonLexer
from pygments.lexers.shell import BashLexer
from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)

_CONSTANT = re.compile(r'^_?[A-Z][A-Z0-9_]+$')
# Names the lexer already classifies precisely; only plain ``Name`` is refined.
_REFINED = {Name, Name.Other}


class VSCodePythonLexer(PythonLexer):
    """PythonLexer with ``Name`` split into call / constant / keyword argument."""

    def get_tokens_unprocessed(self, text, stack=('root',)):
        tokens = list(super().get_tokens_unprocessed(text, stack))
        # The next non-blank token value after each position, in one backward pass --
        # viewcode renders whole modules, so a forward scan per name would be quadratic.
        following_of = [''] * len(tokens)
        following = ''
        for i in range(len(tokens) - 1, -1, -1):
            following_of[i] = following
            if tokens[i][2].strip():
                following = tokens[i][2]
        depth = 0
        for i, (pos, token, value) in enumerate(tokens):
            if token in Punctuation or token in Operator:
                depth += value.count('(') - value.count(')')
            if token in _REFINED:
                following = following_of[i]
                if following.startswith('('):
                    token = Name.Function
                elif following == '=' and depth > 0:
                    token = Name.Variable
                elif _CONSTANT.match(value):
                    token = Name.Constant
            yield pos, token, value


class VSCodeBashLexer(BashLexer):
    """BashLexer with the command word and ``-flags`` told apart from arguments.

    BashLexer emits every bare word as ``Text``, so ``pip install micm-nlp --config x``
    is one colour. VS Code's shell grammar colours the command; this marks the first
    word of each command as ``Name.Function`` and words starting with ``-`` as
    ``Name.Constant``. A command starts at the beginning, after a newline that is not
    a ``\\`` continuation, and after ``;``, ``|``, ``&``, ``&&``, ``||`` or a keyword.
    """

    def get_tokens_unprocessed(self, text, stack=('root',)):
        expect_command = True
        for pos, token, value in super().get_tokens_unprocessed(text, stack):
            if token is Text and value.strip():
                if expect_command:
                    token = Name.Function
                    expect_command = False
                elif value.startswith('-'):
                    token = Name.Constant
            elif token in Name.Builtin and expect_command:
                expect_command = False
            elif token in Keyword or token in Punctuation or value in ('&&', '||'):
                expect_command = True
            elif token in String.Escape and value.startswith('\\\n'):
                pass
            elif '\n' in value:
                expect_command = True
            yield pos, token, value


class OneDarkProDarker(Style):
    """VS Code's One Dark Pro Darker."""

    name = 'one-dark-pro-darker'
    background_color = '#1b1e23'
    highlight_color = '#2c313a'
    line_number_color = '#495162'

    styles = {
        Text: '#abb2bf',
        Error: '#e06c75',
        Comment: 'italic #7f848e',
        Keyword: '#c678dd',
        Keyword.Constant: '#d19a66',
        Keyword.Type: '#e5c07b',
        Operator: '#56b6c2',
        Operator.Word: '#c678dd',
        Punctuation: '#abb2bf',
        Name: '#abb2bf',
        Name.Namespace: '#abb2bf',
        Name.Class: '#e5c07b',
        Name.Function: '#61afef',
        Name.Function.Magic: '#56b6c2',
        Name.Decorator: '#61afef',
        Name.Builtin: '#56b6c2',
        Name.Builtin.Pseudo: 'italic #e06c75',
        Name.Exception: '#e5c07b',
        Name.Constant: '#d19a66',
        Name.Variable: '#e06c75',
        Name.Variable.Magic: '#56b6c2',
        Name.Attribute: '#e06c75',
        Name.Tag: '#e06c75',
        Name.Label: '#e06c75',
        Literal: '#98c379',
        String: '#98c379',
        String.Escape: '#56b6c2',
        String.Interpol: '#c678dd',
        String.Affix: '#c678dd',
        Number: '#d19a66',
        Generic.Heading: 'bold #e06c75',
        Generic.Subheading: 'bold #e06c75',
        Generic.Deleted: '#e06c75',
        Generic.Inserted: '#98c379',
        Generic.Emph: 'italic',
        Generic.Strong: 'bold',
        Generic.Prompt: '#7f848e',
        Generic.Output: '#abb2bf',
    }


class GitHubLight(Style):
    """VS Code's GitHub Light Default."""

    name = 'github-light'
    background_color = '#f6f8fa'
    highlight_color = '#fff8c5'
    line_number_color = '#8c959f'

    styles = {
        Text: '#1f2328',
        Error: '#cf222e',
        Comment: 'italic #6e7781',
        Keyword: '#cf222e',
        Keyword.Constant: '#0550ae',
        Keyword.Type: '#953800',
        Operator: '#0550ae',
        Operator.Word: '#cf222e',
        Punctuation: '#1f2328',
        Name: '#1f2328',
        Name.Namespace: '#1f2328',
        Name.Class: '#953800',
        Name.Function: '#8250df',
        Name.Function.Magic: '#0550ae',
        Name.Decorator: '#8250df',
        Name.Builtin: '#0550ae',
        Name.Builtin.Pseudo: '#953800',
        Name.Exception: '#953800',
        Name.Constant: '#0550ae',
        Name.Variable: '#953800',
        Name.Variable.Magic: '#0550ae',
        Name.Attribute: '#116329',
        Name.Tag: '#116329',
        Name.Label: '#116329',
        Literal: '#0a3069',
        String: '#0a3069',
        String.Escape: '#0550ae',
        String.Interpol: '#cf222e',
        String.Affix: '#cf222e',
        Number: '#0550ae',
        Generic.Heading: 'bold #0550ae',
        Generic.Subheading: 'bold #0550ae',
        Generic.Deleted: '#82071e',
        Generic.Inserted: '#116329',
        Generic.Emph: 'italic',
        Generic.Strong: 'bold',
        Generic.Prompt: '#6e7781',
        Generic.Output: '#1f2328',
    }


def setup(app):
    for alias in ('python', 'python3', 'py'):
        app.add_lexer(alias, VSCodePythonLexer)
    for alias in ('bash', 'sh', 'shell', 'zsh'):
        app.add_lexer(alias, VSCodeBashLexer)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
