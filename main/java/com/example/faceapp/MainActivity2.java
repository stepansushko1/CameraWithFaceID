package com.example.faceapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListAdapter;
import android.widget.ListView;

import java.util.ArrayList;
import java.util.List;

import android.R.layout;
import android.widget.TextView;

import com.google.firebase.crashlytics.buildtools.reloc.org.apache.commons.logging.Log;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class MainActivity2 extends AppCompatActivity {
    Button view_button;
    TextView text;
    List<String> lst = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        view_button = findViewById(R.id.button6);
        text = findViewById(R.id.textView2);

        view_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                FirebaseDatabase.getInstance().getReference().addValueEventListener(new ValueEventListener() {
                    @Override
                    public void onDataChange(@NonNull DataSnapshot snapshot) {
                        lst.clear();
                        for (DataSnapshot snp: snapshot.getChildren()){
//                            for (DataSnapshot snp1: snp.getChildren()){
                                lst.add(snp.getValue().toString());
//                            }
                        }
                        for (int i =0; i<lst.size();i++){
                            text.setText(lst.get(i)+"\n");}
                    }


                    @Override
                    public void onCancelled(@NonNull DatabaseError error) {
                    }
                });
            }
        });
    }
    public String transform_info(String info){
        String[] result = info.split("Name=");
        return result.toString();
    }

}