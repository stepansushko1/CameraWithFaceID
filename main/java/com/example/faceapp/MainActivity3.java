package com.example.faceapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.google.firebase.database.FirebaseDatabase;

import org.w3c.dom.Text;

import java.util.UUID;

public class MainActivity3 extends AppCompatActivity {

    Button addu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);

        addu = findViewById(R.id.add_user_button);

        addu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                add_user_db();
            }
        });
    }
    public void add_user_db(){
        final EditText uname = (EditText) findViewById(R.id.username);
        final EditText usurname = (EditText) findViewById(R.id.usersurname);

        String temp_name = uname.getText().toString();
        String temp_surname = usurname.getText().toString();

        String uniqueID = UUID.randomUUID().toString();
        FirebaseDatabase.getInstance().getReference().child("User").push().child(uniqueID);

        FirebaseDatabase.getInstance().getReference().child("User").child(uniqueID).push().child("Name").setValue(temp_name+" "+temp_surname);
        FirebaseDatabase.getInstance().getReference().child("User").child(uniqueID).push().child("Name").setValue("Testing value");
    }
}