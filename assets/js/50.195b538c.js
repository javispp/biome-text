(window.webpackJsonp=window.webpackJsonp||[]).push([[50],{376:function(a,t,e){"use strict";e.r(t);var s=e(26),v=Object(s.a)({},(function(){var a=this,t=a.$createElement,e=a._self._c||t;return e("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[e("h1",{attrs:{id:"biome-text-vocabulary"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-vocabulary"}},[a._v("#")]),a._v(" biome.text.vocabulary "),e("Badge",{attrs:{text:"Module"}})],1),a._v(" "),e("div"),a._v(" "),e("p",[a._v("Manages vocabulary tasks and fetches vocabulary information")]),a._v(" "),e("p",[a._v("Provides utilities for getting information from a given vocabulary.")]),a._v(" "),e("p",[a._v('Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.')]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"get-labels"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#get-labels"}},[a._v("#")]),a._v(" get_labels "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("get_labels")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> List[str]")]),a._v("\n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Gets list of labels in the vocabulary")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")])]),a._v(" "),e("h2",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("labels")])]),a._v(" : "),e("code",[a._v("List[str]")])]),a._v(" "),e("dd",[a._v("A list of label strings")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"label-for-index"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#label-for-index"}},[a._v("#")]),a._v(" label_for_index "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("label_for_index")]),a._v(" ("),a._v("\n  vocab: allennlp.data.vocabulary.Vocabulary,\n  idx: int,\n)  -> str\n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Gets label string for a label "),e("code",[a._v("int")]),a._v(" id")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")]),a._v(" "),e("dt",[e("strong",[e("code",[a._v("idx")])]),a._v(" : "),e("code",[a._v("`int")])]),a._v(" "),e("dd",[a._v("the token index")])]),a._v(" "),e("h2",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("label")])]),a._v(" : "),e("code",[a._v("str")])]),a._v(" "),e("dd",[a._v("The string for a label id")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"index-for-label"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#index-for-label"}},[a._v("#")]),a._v(" index_for_label "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("index_for_label")]),a._v(" ("),a._v("\n  vocab: allennlp.data.vocabulary.Vocabulary,\n  label: str,\n)  -> int\n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Gets the label "),e("code",[a._v("int")]),a._v(" id for label string")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : ``allennlp.data.Vocabulary```")]),a._v(" "),e("dd",[a._v(" ")]),a._v(" "),e("dt",[e("strong",[e("code",[a._v("label")])]),a._v(" : "),e("code",[a._v("str")])]),a._v(" "),e("dd",[a._v("the label")])]),a._v(" "),e("h2",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("label_idx")])]),a._v(" : "),e("code",[a._v("int")])]),a._v(" "),e("dd",[a._v("The label id for label string")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"get-index-to-labels-dictionary"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#get-index-to-labels-dictionary"}},[a._v("#")]),a._v(" get_index_to_labels_dictionary "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("get_index_to_labels_dictionary")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> Dict[int, str]")]),a._v("\n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Gets a dictionary for turning label "),e("code",[a._v("int")]),a._v(" ids into label strings")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")])]),a._v(" "),e("h2",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("labels")])]),a._v(" : "),e("code",[a._v("Dict[int, str]")])]),a._v(" "),e("dd",[a._v("A dictionary to get fetch label strings from ids")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"words-vocab-size"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#words-vocab-size"}},[a._v("#")]),a._v(" words_vocab_size "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("words_vocab_size")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> int")]),a._v("\n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Fetches the vocabulary size for the "),e("code",[a._v("words")]),a._v(" namespace")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")])]),a._v(" "),e("h2",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("size")])]),a._v(" : "),e("code",[a._v("int")])]),a._v(" "),e("dd",[a._v("The vocabulary size for the words namespace")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"extend-labels"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#extend-labels"}},[a._v("#")]),a._v(" extend_labels "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("extend_labels")]),a._v(" ("),a._v("\n  vocab: allennlp.data.vocabulary.Vocabulary,\n  labels: List[str],\n) \n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Adds a list of label strings to the vocabulary")]),a._v(" "),e("p",[a._v("Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")]),a._v(" "),e("dt",[e("strong",[e("code",[a._v("labels")])]),a._v(" : "),e("code",[a._v("List[str]")])]),a._v(" "),e("dd",[a._v("A list of strings containing the labels to add to an existing vocabulary")])])]),a._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"set-labels"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#set-labels"}},[a._v("#")]),a._v(" set_labels "),e("Badge",{attrs:{text:"Function"}})],1),a._v("\n")]),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("set_labels")]),a._v(" ("),a._v("\n  vocab: allennlp.data.vocabulary.Vocabulary,\n  new_labels: List[str],\n) \n")]),a._v("\n")])])]),a._v(" "),e("dd",[e("p",[a._v("Resets the labels in the vocabulary with a given labels string list")]),a._v(" "),e("h2",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("dl",[e("dt",[e("strong",[e("code",[a._v("vocab")])]),a._v(" : "),e("code",[a._v("allennlp.data.Vocabulary")])]),a._v(" "),e("dd",[a._v(" ")]),a._v(" "),e("dt",[e("strong",[e("code",[a._v("new_labels")])]),a._v(" : "),e("code",[a._v("List[str]")])]),a._v(" "),e("dd",[a._v("The label strings to add to the vocabulary")])])])])}),[],!1,null,null,null);t.default=v.exports}}]);