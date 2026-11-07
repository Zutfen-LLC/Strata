#!/usr/bin/env python3
"""Static wiring control only; does not construct or run a selected model."""
from pathlib import Path
root = Path(__file__).resolve().parents[2] / "src/core"
def has(file, *needles):
    text = (root / file).read_text(encoding="utf-8")
    for needle in needles:
        assert needle in text, f"{file}: missing {needle}"
has("portable_native_model_owner.hpp", 'b.entry,method,gdn_method,ab_method,selection', 'b.ple,method,gdn_method,ab_method,selection', 'b.gdn,method,gdn_method,ab_method,selection', 'b.qsa,method,selection', 'b.terminal,method,selection', 'attention_hc_read_methods()', 'ffn_hc_read_methods()', 'PLE selected host budget')
has("portable_native_layer0_owner.hpp", 'create_selected(0,backend,ah', 'create_selected(0,backend,f', 'selection,true')
has("portable_native_ple1_owner.hpp", 'b.ple,pp,selection', 'fp.c_str(),b.ffn_caps,selection')
has("portable_native_ordinary_gdn_factory.hpp", 'create_selected(layer,backend,hc', 'b.hc_caps,selection,true')
has("portable_native_qsa_owner.hpp", 'create_selected(layer,b', 'caps.hc_caps,selection,true')
has("portable_native_ordinary_layer_owner.hpp", 'b.ffn_caps,selection')
has("portable_native_qsa_layer_owner.hpp", 'caps.ffn_caps,selection')
has("portable_native_terminal_owner.hpp", 'b.qsa,selection', 'b.terminal_caps,selection', 'PortableHead::create(geo,hv,opts,error)')
has("portable_native_ffn_composition.hpp", 'create_selected(source_layer(source)', 'hc_read_method()const{return hc_->read_method();}')
print("HC owner selector source wiring only; no selected model construction")
