(window.webpackJsonp=window.webpackJsonp||[]).push([[45],{382:function(t,e,a){"use strict";a.r(e);var s=a(26),n=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-text-cleaning"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-text-cleaning"}},[t._v("#")]),t._v(" biome.text.text_cleaning "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("dl",[a("h2",{attrs:{id:"biome.text.text_cleaning.TextCleaning"}},[t._v("TextCleaning "),a("Badge",{attrs:{text:"Class"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[t._v("    "),a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextCleaning")]),t._v(" ()"),t._v("\n    ")])])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Base class for text cleaning processors")])]),t._v(" "),a("h3",[t._v("Ancestors")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),a("h3",[t._v("Class variables")]),t._v(" "),a("dl",[a("dt",{attrs:{id:"biome.text.text_cleaning.TextCleaning.default_implementation"}},[a("code",{staticClass:"name"},[t._v("var "),a("span",{staticClass:"ident"},[t._v("default_implementation")]),t._v(" : str")])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"})])])]),t._v(" "),a("h2",{attrs:{id:"biome.text.text_cleaning.TextCleaningRule"}},[t._v("TextCleaningRule "),a("Badge",{attrs:{text:"Class"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[t._v("    "),a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("TextCleaningRule")]),t._v(" (func: Callable[[str], str])"),t._v("\n    ")])])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Registers a function as a rule for the default text cleaning implementation")]),t._v(" "),a("p",[t._v("Use the decorator "),a("code",[t._v("@TextCleaningRule")]),t._v(" for creating custom text cleaning and pre-processing rules.")]),t._v(" "),a("p",[t._v("An example function to strip spaces (already included in the default "),a("code",[a("a",{attrs:{title:"biome.text.text_cleaning.TextCleaning",href:"#biome.text.text_cleaning.TextCleaning"}},[t._v("TextCleaning")])]),t._v(" processor):")]),t._v(" "),a("pre",[a("code",{staticClass:"python"},[t._v("@TextCleaningRule\ndef strip_spaces(text: str) -> str:\n    return text.strip()\n")])]),t._v(" "),a("h1",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("pre",[a("code",[t._v("func: <code>Callable\\[\\[str]</code>\n    The function to register\n")])])]),t._v(" "),a("dl",[a("h3",{attrs:{id:"biome.text.text_cleaning.TextCleaningRule.registered_rules"}},[t._v("registered_rules "),a("Badge",{attrs:{text:"Static method"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("registered_rules")]),t._v("("),a("span",[t._v(") -> Dict[str, Callable[[str], str]]")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Registered rules dictionary")])])])])]),t._v(" "),a("h2",{attrs:{id:"biome.text.text_cleaning.DefaultTextCleaning"}},[t._v("DefaultTextCleaning "),a("Badge",{attrs:{text:"Class"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[t._v("    "),a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("DefaultTextCleaning")]),t._v(" (rules: List[str] = None)"),t._v("\n    ")])])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Defines rules that can be applied to the text before it gets tokenized.")]),t._v(" "),a("p",[t._v("Each rule is a simple python function that receives and returns a "),a("code",[t._v("str")]),t._v(".")]),t._v(" "),a("h1",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),a("pre",[a("code",[t._v("rules: <code>List\\[str]</code>\n    A list of registered rule method names to be applied to text inputs\n")])])]),t._v(" "),a("h3",[t._v("Ancestors")]),t._v(" "),a("ul",{staticClass:"hlist"},[a("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),a("li",[t._v("allennlp.common.from_params.FromParams")])])])])])}),[],!1,null,null,null);e.default=n.exports}}]);