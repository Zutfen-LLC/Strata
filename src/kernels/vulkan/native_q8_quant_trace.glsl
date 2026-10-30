    // Q8_TRACE_BEGIN: stores only; original arithmetic and controls stay intact.
    uint trace_slot=i==36u?0u:i==118u?1u:i==1046u?2u:i==4479u?3u:4u;
    if(p.mode==0u && trace_slot<4u) {
        hidden[p.hidden_bias+trace_slot*4u+0u]=maxima[lane];
        hidden[p.hidden_bias+trace_slot*4u+1u]=d;
        hidden[p.hidden_bias+trace_slot*4u+2u]=ratio;
        hidden[p.hidden_bias+trace_slot*4u+3u]=rounded;
    }
    // Q8_TRACE_END
